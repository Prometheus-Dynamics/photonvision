//! Eidos AprilTag detection with tracking, per-tag poses and (optionally) one multi-tag camera
//! pose, for one camera. Everything here runs on the frame's Styx lease; nothing is copied.

use std::collections::HashSet;
use std::time::Duration;

use eidos_aruco::ArucoDictionaryKind;
use eidos_calibration::{
    CalibrationStatus, CameraCalibration, CameraIntrinsics, DistortionCoefficients, KnownTag,
    KnownTagPoses, LensModel, MultiTagPose, MultiTagPoseConfig, MultiTagPoseSolver, PixelCoordinate,
    PnpRefineConfig, Point3, Se3Transform, TagCorners, project_camera_point, solve_square_pnp,
    square_object_corners, transform_point, undistort_points,
};
use eidos_core::FrameLease;
use styx::core::prelude::ProgressWait;
use eidos_detectors::aruco::{TrackedTagArgs, TrackedTagDetector, TrackedTagOutput, TrackedTagSlices};
use eidos_core::catalog::OpParams;
use eidos_detectors::catalog::TrackedDetectorParams;

use crate::config::{CalibrationConfig, PipelineConfig};
use crate::geometry::{eidos_reference_from_tag, pose_array, se3_from_wpilib, wpilib_camera_from_optical};
use crate::pack::{MultiRecord, TargetRecord};

/// What one frame produced.
#[derive(Default)]
pub struct Detections {
    pub targets: Vec<TargetRecord>,
    pub multi: MultiRecord,
    /// What the processed stream draws for each target (same order as `targets`).
    pub overlays: Vec<Overlay>,
}

/// One tag as the processed stream draws it, in full-frame pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Overlay {
    pub id: i32,
    /// Refined corners, marker order (`[u0, v0, ..., u3, v3]`).
    pub corners: [f64; 8],
    /// In 3D mode: the corners of the box's far face, one tag edge out of the tag towards the
    /// camera (PhotonVision's 3D box), projected through the calibration. Same order as `corners`.
    pub top: Option<[f64; 8]>,
}

/// The far face of the 3D box on a tag of edge `side` at `camera_from_tag` (Eidos tag frame: z
/// into the tag, so the box rises along -z), projected to distorted pixels.
pub fn box_top(camera_from_tag: &Se3Transform, side: f64, calibration: &CameraCalibration) -> Option<[f64; 8]> {
    let mut out = [0.0; 8];
    for (i, corner) in square_object_corners(side).into_iter().enumerate() {
        let lifted = Point3 { x: corner.x, y: corner.y, z: -side };
        let p = transform_point(camera_from_tag, lifted);
        if p.z <= 1e-6 {
            return None;
        }
        let px = project_camera_point(p, *calibration)?;
        out[2 * i] = px.u;
        out[2 * i + 1] = px.v;
    }
    Some(out)
}

pub struct Detector {
    config: PipelineConfig,
    tracked: TrackedTagDetector,
    out: TrackedTagOutput,
    /// A frame delivered in slices (rows still arriving), reused across frames.
    slices: TrackedTagSlices,
    calibration: Option<CameraCalibration>,
    solver: Option<MultiTagPoseSolver>,
    multi_out: MultiTagPose,
    observations: Vec<TagCorners>,
}

fn dictionary(family: &str) -> ArucoDictionaryKind {
    match family {
        "tag16h5" => ArucoDictionaryKind::AprilTag16h5,
        _ => ArucoDictionaryKind::AprilTag36h11,
    }
}

pub fn calibration_from(config: &CalibrationConfig) -> CameraCalibration {
    CameraCalibration {
        intrinsics: CameraIntrinsics {
            fx: config.fx,
            fy: config.fy,
            cx: config.cx,
            cy: config.cy,
            skew: config.skew,
        },
        distortion: DistortionCoefficients {
            k1: config.k1,
            k2: config.k2,
            k3: config.k3,
            k4: config.k4,
            k5: config.k5,
            k6: config.k6,
            p1: config.p1,
            p2: config.p2,
        },
        // The same iteration count the Eidos calibration tests use.
        undistort_iters: 5,
        lens_model: if config.fisheye { LensModel::Fisheye } else { LensModel::Pinhole },
    }
}

impl Detector {
    pub fn new(config: &PipelineConfig) -> Result<Self, String> {
        config.validate()?;
        let mut params = TrackedDetectorParams::from_params(&[]);
        params.dictionary = dictionary(&config.family);
        params.full_search_every = config.full_search_every.max(1);
        let tracked = params.detector();

        let calibration = config.calibration.as_ref().map(calibration_from);
        let solver = if config.multi_tag && !config.field_tags.is_empty() {
            let known = config
                .field_tags
                .iter()
                .map(|tag| KnownTag {
                    id: tag.id,
                    side: config.tag_size_for(tag.id),
                    reference_from_tag: eidos_reference_from_tag(se3_from_wpilib(
                        tag.x, tag.y, tag.z, tag.qw, tag.qx, tag.qy, tag.qz,
                    )),
                })
                .collect();
            Some(MultiTagPoseSolver::new(
                KnownTagPoses::new(known),
                MultiTagPoseConfig::default(),
            ))
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            tracked,
            out: TrackedTagOutput::default(),
            slices: TrackedTagSlices::default(),
            calibration,
            solver,
            multi_out: MultiTagPose::default(),
            observations: Vec::new(),
        })
    }

    /// Detect every tag in `frame` (its half-size pyramid level when it has one).
    pub fn process(&mut self, frame: &FrameLease) -> Result<Detections, String> {
        let level = frame.pyramid_level(1).unwrap_or(frame);
        self.tracked
            .apply_into(TrackedTagArgs { frame, level }, &mut self.out)
            .map_err(|e| format!("tag detection failed: {e:?}"))?;
        self.results()
    }

    /// [`Self::process`] on a frame that may arrive in slices (Styx's sliced delivery: the
    /// lease comes at the first slice, its rows still arriving). The detector works on the rows
    /// already final, then on the rest once the frame completes, so little is left after the
    /// last row; the detections are the same as on the complete frame. `Ok(None)`: the frame
    /// failed or did not complete within `timeout`, and is dropped.
    pub fn process_sliced(&mut self, frame: &FrameLease, timeout: Duration) -> Result<Option<Detections>, String> {
        let Some(progress) = frame.progress() else {
            return self.process(frame).map(Some);
        };
        let level = frame.pyramid_level(1).unwrap_or(frame);
        let args = TrackedTagArgs { frame, level };
        // The level's own rows when it is delivered in slices too; otherwise none until the
        // frame completes.
        let level_rows = || level.progress().map_or(0, |p| p.rows() as usize);
        let early = self
            .tracked
            .begin_frame(args, &mut self.slices)
            .and_then(|()| {
                self.tracked
                    .rows_ready(args, progress.rows() as usize, level_rows(), &mut self.slices)
            });
        if let Err(e) = early {
            self.tracked.abandon(&mut self.slices);
            return Err(format!("tag detection failed: {e:?}"));
        }
        if progress.wait_complete(timeout) != ProgressWait::Complete {
            self.tracked.abandon(&mut self.slices);
            return Ok(None);
        }
        let height = |f: &FrameLease| f.meta().format.resolution.height.get() as usize;
        self.tracked
            .rows_ready(args, height(frame), height(level), &mut self.slices)
            .and_then(|()| self.tracked.finish(args, &mut self.slices, &mut self.out))
            .map_err(|e| format!("tag detection failed: {e:?}"))?;
        self.results().map(Some)
    }

    /// The detections of the frame just detected (`self.out`), with poses when calibrated.
    fn results(&mut self) -> Result<Detections, String> {

        let kept = &self.out.validated.kept.items;
        let refined = &self.out.refined.items;
        let mut targets = Vec::with_capacity(kept.len());
        let mut overlays = Vec::with_capacity(kept.len());
        let solve = self.config.solve_pnp;
        let mut corners_px: Vec<[PixelCoordinate; 4]> = Vec::with_capacity(kept.len());

        for (i, det) in kept.iter().enumerate() {
            let Some(quad) = refined.get(i).or_else(|| self.out.corners.items.get(i)) else {
                continue;
            };
            let px = quad.corners.map(|p| PixelCoordinate {
                u: f64::from(p.x),
                v: f64::from(p.y),
            });
            let flat = [
                px[0].u, px[0].v, px[1].u, px[1].v, px[2].u, px[2].v, px[3].u, px[3].v,
            ];
            let mut top = None;
            let (best, alt, ambiguity, rms_best, rms_alt) = match &self.calibration {
                Some(calibration) if solve => {
                    let side = self.config.tag_size_for(det.id);
                    match solve_square_pnp(&px, side, calibration, PnpRefineConfig::default()) {
                        Some(solution) => {
                            if self.config.draw {
                                top = box_top(&solution.best.camera_from_object, side, calibration);
                            }
                            (
                                pose_array(&solution.best.camera_from_object),
                                solution.alternative.map(|a| pose_array(&a.camera_from_object)),
                                solution.ambiguity,
                                solution.best.rms_px,
                                solution.alternative.map_or(0.0, |a| a.rms_px),
                            )
                        }
                        None => ([f64::NAN; 7], None, 0.0, 0.0, 0.0),
                    }
                }
                _ => ([f64::NAN; 7], None, 0.0, 0.0, 0.0),
            };
            // The tag centre, undistorted into the same intrinsics (pinhole): what PhotonVision's
            // yaw and pitch use.
            let centre_px = PixelCoordinate {
                u: px.iter().map(|p| p.u).sum::<f64>() / 4.0,
                v: px.iter().map(|p| p.v).sum::<f64>() / 4.0,
            };
            let center = match self.calibration.as_ref() {
                Some(calibration) => undistort_points(&[centre_px], calibration, calibration.intrinsics)
                    .first()
                    .copied()
                    .flatten()
                    .map_or([f64::NAN; 2], |p| [p.u, p.v]),
                None => [f64::NAN; 2],
            };
            targets.push(TargetRecord {
                id: det.id as i32,
                hamming: i32::from(det.hamming_distance),
                used_in_multi: false,
                corners: flat,
                best,
                alt,
                ambiguity,
                rms_best_px: rms_best,
                rms_alt_px: rms_alt,
                center,
            });
            overlays.push(Overlay { id: det.id as i32, corners: flat, top });
            corners_px.push(px);
        }

        let mut multi = MultiRecord::default();
        if let (Some(solver), Some(calibration)) = (self.solver.as_mut(), self.calibration.as_ref())
            && solve
            && calibration.status() != CalibrationStatus::Uncalibrated
        {
            self.observations.clear();
            for (target, px) in targets.iter().zip(&corners_px) {
                self.observations.push(TagCorners {
                    id: target.id as u32,
                    corners: *px,
                });
            }
            if solver.solve_into(&self.observations, calibration, None, &mut self.multi_out)
                && self.multi_out.valid
            {
                let used: HashSet<u32> = self
                    .multi_out
                    .tags
                    .iter()
                    .filter(|t| t.inlier)
                    .map(|t| t.id)
                    .collect();
                for target in &mut targets {
                    target.used_in_multi = used.contains(&(target.id as u32));
                }
                multi = MultiRecord {
                    valid: true,
                    inlier_tags: self.multi_out.inlier_tags as i32,
                    rms_px: self.multi_out.rms_px,
                    ambiguity: self.multi_out.ambiguity,
                    pose: pose_array(&wpilib_camera_from_optical(self.multi_out.reference_from_camera)),
                };
            }
        }

        Ok(Detections { targets, multi, overlays })
    }
}
