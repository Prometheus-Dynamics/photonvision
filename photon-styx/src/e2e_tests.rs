//! End to end on a synthetic frame: a tag36h11 rendered at a known pose through the Raze's
//! fisheye calibration (the one PhotonVision stored for 1280x800), detected by the same
//! `Detector` the camera session runs, its pose and 3D box checked against the truth.

use eidos_aruco::{ArucoDictionaryKind, BitGrid, dictionary};
use eidos_calibration::{
    CameraCalibration, PixelCoordinate, RotationMatrix3, Se3Transform, Translation3, unproject_pixel,
};
use eidos_core::{BufferPool, FrameLease};
use styx::prelude::{ColorSpace, FourCc, MediaFormat, Resolution};

use crate::config::{CalibrationConfig, PipelineConfig};
use crate::detect::{Detector, box_top, calibration_from};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 800;
const SIDE: f64 = 0.1651;

/// The Raze's calibration at 1280x800 (OpenCV fisheye, from PhotonVision's database).
fn raze_calibration() -> CalibrationConfig {
    CalibrationConfig {
        fx: 568.979_855_850_873_5,
        fy: 570.845_616_585_421_8,
        cx: 612.743_629_259_751_2,
        cy: 332.741_894_738_398_23,
        skew: -0.944_043_752_408_552_4,
        k1: -0.038_266_971_912_260_22,
        k2: 0.131_653_727_406_827_82,
        k3: -0.179_826_912_161_370_62,
        k4: 0.085_773_209_352_616_9,
        k5: 0.0,
        k6: 0.0,
        p1: 0.0,
        p2: 0.0,
        fisheye: true,
    }
}

fn rot_x(a: f64) -> [[f64; 3]; 3] {
    let (s, c) = a.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn rot_y(a: f64) -> [[f64; 3]; 3] {
    let (s, c) = a.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// A tag 1.4 m ahead, right of and above the axis, turned 25 degrees about the camera's y and
/// tilted 10 degrees about its x (Eidos tag frame: x right, y down, z into the tag).
fn truth() -> Se3Transform {
    Se3Transform {
        rotation: RotationMatrix3 {
            m: crate::geometry::mat_mul(rot_y(25f64.to_radians()), rot_x(-10f64.to_radians())),
        },
        translation: Translation3 { x: 0.35, y: -0.12, z: 1.4 },
    }
}

/// Grey value of the scene at tag-plane point (`x`, `y`): the tag (black border, 6x6 bits) in a
/// one-cell white quiet zone, on a mid-grey wall.
fn scene_value(grid: &BitGrid, x: f64, y: f64) -> f64 {
    let cell = SIDE / 8.0;
    let (cx, cy) = (((x + SIDE / 2.0) / cell).floor(), ((y + SIDE / 2.0) / cell).floor());
    if !(-1.0..9.0).contains(&cx) || !(-1.0..9.0).contains(&cy) {
        return 120.0;
    }
    let (cx, cy) = (cx as i64, cy as i64);
    if cx == -1 || cy == -1 || cx == 8 || cy == 8 {
        return 230.0;
    }
    if cx == 0 || cy == 0 || cx == 7 || cy == 7 {
        return 20.0;
    }
    if grid.bit(cx as usize - 1, cy as usize - 1) { 230.0 } else { 20.0 }
}

/// The frame a camera with `calibration` sees of tag `id` at `camera_from_tag` (2x2 supersampled),
/// with its half-size pyramid level.
fn render(id: u32, camera_from_tag: &Se3Transform, calibration: &CameraCalibration, pool: &BufferPool) -> FrameLease {
    let grid = BitGrid::from_marker(dictionary(ArucoDictionaryKind::AprilTag36h11), id, 0).expect("tag grid");
    let r = camera_from_tag.rotation.m;
    let t = camera_from_tag.translation;
    // The tag plane's normal (tag z) and a point on it, in the camera frame.
    let n = [r[0][2], r[1][2], r[2][2]];
    let n_dot_t = n[0] * t.x + n[1] * t.y + n[2] * t.z;
    let mut pixels = vec![120u8; (WIDTH * HEIGHT) as usize];
    for (v, row) in pixels.chunks_mut(WIDTH as usize).enumerate() {
            for (u, value) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for (du, dv) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                    let px = PixelCoordinate { u: u as f64 + du - 0.5, v: v as f64 + dv - 0.5 };
                    let Some(ray) = unproject_pixel(px, calibration) else {
                        sum += 120.0;
                        continue;
                    };
                    let d = [ray.x, ray.y, 1.0];
                    let denom = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
                    let s = n_dot_t / denom;
                    if !(s.is_finite() && s > 0.0) {
                        sum += 120.0;
                        continue;
                    }
                    // The hit point in tag coordinates: R^T (s d - t).
                    let p = [s * d[0] - t.x, s * d[1] - t.y, s * d[2] - t.z];
                    let x = r[0][0] * p[0] + r[1][0] * p[1] + r[2][0] * p[2];
                    let y = r[0][1] * p[0] + r[1][1] * p[1] + r[2][1] * p[2];
                    sum += scene_value(&grid, x, y);
                }
                *value = (sum / 4.0).round() as u8;
            }
    }
    let format = MediaFormat::new(
        FourCc::GREY,
        Resolution::new(WIDTH, HEIGHT).expect("size"),
        ColorSpace::Unknown,
    );
    let frame = FrameLease::from_visible_bytes(format, 0, &pixels).expect("frame");
    frame.with_box_pyramid_in(1, 1, pool).expect("pyramid")
}

fn quaternion_angle(a: &[f64; 7], b: &[f64; 7]) -> f64 {
    let dot = (a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6]).abs().min(1.0);
    2.0 * dot.acos()
}

#[test]
fn detects_and_solves_a_tag_through_the_raze_fisheye_calibration() {
    let cal_config = raze_calibration();
    let calibration = calibration_from(&cal_config);
    let pool = BufferPool::lazy(0, 4);
    let truth = truth();
    let frame = render(7, &truth, &calibration, &pool);

    let config = PipelineConfig {
        family: "tag36h11".into(),
        tag_size_m: SIDE,
        calibration: Some(cal_config),
        multi_tag: false,
        field_tags: Vec::new(),
        full_search_every: 8,
        processed_stream: true,
        draw: true,
        solve_pnp: true,
        stream_divisor: 1,
    };
    let mut detector = Detector::new(&config).expect("detector");
    let found = detector.process(&frame).expect("detection");
    assert_eq!(found.targets.len(), 1, "one tag");
    let target = &found.targets[0];
    assert_eq!(target.id, 7);

    let expected = crate::geometry::pose_array(&truth);
    let (dx, dy, dz) = (
        target.best[0] - expected[0],
        target.best[1] - expected[1],
        target.best[2] - expected[2],
    );
    let position_error = (dx * dx + dy * dy + dz * dz).sqrt();
    let angle_error = quaternion_angle(&target.best, &expected).to_degrees();
    eprintln!("pose error: {:.2} mm, {:.3} deg, rms {:.3} px", position_error * 1e3, angle_error, target.rms_best_px);
    assert!(position_error < 0.01, "position off by {position_error} m: {:?} vs {expected:?}", target.best);
    assert!(angle_error < 2.0, "rotation off by {angle_error} deg");
    assert!(target.rms_best_px < 1.0, "reprojection rms {}", target.rms_best_px);

    // The 3D box's far face, from the truth, where the overlay draws it (within a pixel or two).
    let overlay = &found.overlays[0];
    let top = overlay.top.expect("3D box in 3D mode");
    let true_top = box_top(&truth, SIDE, &calibration).expect("box");
    for i in 0..8 {
        assert!((top[i] - true_top[i]).abs() < 3.0, "box corner {i}: {} vs {}", top[i], true_top[i]);
    }

    // In 2D mode: no pose and no box.
    let mut flat = Detector::new(&PipelineConfig { solve_pnp: false, ..config.clone() }).expect("detector");
    let found = flat.process(&frame).expect("detection");
    assert!(found.targets[0].best[0].is_nan());
    assert!(found.overlays[0].top.is_none());
}
