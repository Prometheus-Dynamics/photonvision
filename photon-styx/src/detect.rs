//! Eidos AprilTag detection with tracking, per-tag poses and (optionally) one multi-tag camera
//! pose, for one camera. Everything here runs on the frame's Styx lease; nothing is copied.

use std::collections::HashSet;

use eidos_aruco::ArucoDictionaryKind;
use eidos_calibration::{
    CalibrationStatus, CameraCalibration, CameraIntrinsics, DistortionCoefficients, KnownTag,
    KnownTagPoses, LensModel, MultiTagPose, MultiTagPoseConfig, MultiTagPoseSolver, PixelCoordinate,
    PnpRefineConfig, TagCorners, solve_square_pnp, undistort_points,
};
use eidos_core::FrameLease;
use eidos_detectors::aruco::{TrackedTagArgs, TrackedTagDetector, TrackedTagOutput};
use eidos_core::catalog::OpParams;
use eidos_detectors::catalog::TrackedDetectorParams;

use crate::config::{CalibrationConfig, PipelineConfig};
use crate::geometry::{eidos_reference_from_tag, pose_array, se3_from_wpilib, wpilib_camera_from_optical};
use crate::pack::{MultiRecord, TargetRecord};

/// What one frame produced.
pub struct Detections {
    pub targets: Vec<TargetRecord>,
    pub multi: MultiRecord,
    /// Refined corners of every target, for overlays (same order as `targets`).
    pub quads: Vec<[f64; 8]>,
}

pub struct Detector {
    config: PipelineConfig,
    tracked: TrackedTagDetector,
    out: TrackedTagOutput,
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

        let kept = &self.out.validated.kept.items;
        let refined = &self.out.refined.items;
        let mut targets = Vec::with_capacity(kept.len());
        let mut quads = Vec::with_capacity(kept.len());
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
            let (best, alt, ambiguity, rms_best, rms_alt) = match (&self.calibration, self.config.calibration.is_some()) {
                (Some(calibration), true) => {
                    let side = self.config.tag_size_for(det.id);
                    match solve_square_pnp(&px, side, calibration, PnpRefineConfig::default()) {
                        Some(solution) => (
                            pose_array(&solution.best.camera_from_object),
                            solution.alternative.map(|a| pose_array(&a.camera_from_object)),
                            solution.ambiguity,
                            solution.best.rms_px,
                            solution.alternative.map_or(0.0, |a| a.rms_px),
                        ),
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
            quads.push(flat);
            corners_px.push(px);
        }

        let mut multi = MultiRecord::default();
        if let (Some(solver), Some(calibration)) = (self.solver.as_mut(), self.calibration.as_ref())
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

        Ok(Detections { targets, multi, quads })
    }
}
