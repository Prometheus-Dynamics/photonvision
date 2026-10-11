//! Synthetic cameras: a known field pose projects known tags, and the conversions and solvers
//! must return that pose. The field frame is WPILib (NWU); the camera looks down +x.

use eidos_calibration::{
    CameraCalibration, CameraIntrinsics, DistortionCoefficients, LensModel, MultiTagPose,
    MultiTagPoseConfig, MultiTagPoseSolver, KnownTag, KnownTagPoses, PixelCoordinate,
    PnpRefineConfig, Se3Transform, TagCorners, Translation3, RotationMatrix3, se3_compose,
    se3_inverse, solve_square_pnp, square_object_corners,
};

use crate::detect::calibration_from;
use crate::config::CalibrationConfig;
use crate::geometry::{eidos_reference_from_tag, pose_array, se3_from_wpilib, wpilib_camera_from_optical};

fn calibration() -> CameraCalibration {
    calibration_from(&CalibrationConfig {
        fx: 800.0,
        fy: 800.0,
        cx: 640.0,
        cy: 400.0,
        skew: 0.0,
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        k4: 0.0,
        k5: 0.0,
        k6: 0.0,
        p1: 0.0,
        p2: 0.0,
        fisheye: false,
    })
}

/// Camera at `position` (field frame) looking along +x, optical frame (x right, y down, z fwd).
fn camera_looking_down_x(position: [f64; 3]) -> Se3Transform {
    Se3Transform {
        rotation: RotationMatrix3 {
            m: [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        },
        translation: Translation3 {
            x: position[0],
            y: position[1],
            z: position[2],
        },
    }
}

/// Tag facing the camera at field `(x, y, z)`: WPILib tag frame x points back at the origin,
/// so the tag is rotated 180 degrees about z.
fn facing_back_tag(x: f64, y: f64, z: f64) -> Se3Transform {
    se3_from_wpilib(x, y, z, 0.0, 0.0, 0.0, 1.0)
}

/// The corners a camera at `reference_from_camera` sees of a tag, in Eidos marker order.
fn project(reference_from_camera: &Se3Transform, reference_from_tag: &Se3Transform, side: f64, cal: &CameraCalibration) -> [PixelCoordinate; 4] {
    let camera_from_reference = se3_inverse(*reference_from_camera);
    square_object_corners(side).map(|corner| {
        let in_reference = eidos_calibration::transform_point(reference_from_tag, corner);
        let p = eidos_calibration::transform_point(&camera_from_reference, in_reference);
        PixelCoordinate {
            u: cal.intrinsics.fx * p.x / p.z + cal.intrinsics.cx,
            v: cal.intrinsics.fy * p.y / p.z + cal.intrinsics.cy,
        }
    })
}

fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() < tol, "{what}: {a} vs {b}");
}

fn assert_pose_close(a: &Se3Transform, b: &Se3Transform, tol: f64) {
    for (r, row) in a.rotation.m.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            assert_close(v, b.rotation.m[r][c], tol, "rotation");
        }
    }
    assert_close(a.translation.x, b.translation.x, tol, "x");
    assert_close(a.translation.y, b.translation.y, tol, "y");
    assert_close(a.translation.z, b.translation.z, tol, "z");
}

#[test]
fn single_tag_pose_is_the_tag_in_the_camera_frame() {
    let cal = calibration();
    let side = 0.2;
    let camera = camera_looking_down_x([-1.0, 0.3, 1.2]);
    let field_from_tag = facing_back_tag(5.0, 0.4, 1.0);
    let reference_from_tag = eidos_reference_from_tag(field_from_tag);
    let corners = project(&camera, &reference_from_tag, side, &cal);

    let solution = solve_square_pnp(&corners, side, &cal, PnpRefineConfig::default()).expect("pose");
    // The tag is 6 m straight ahead, 0.7 m to the right of the optical axis, 0.2 m below it.
    let truth = se3_compose(se3_inverse(camera), reference_from_tag);
    assert_pose_close(&solution.best.camera_from_object, &truth, 1e-3);
    assert!(solution.best.rms_px < 1e-3);
}

#[test]
fn wpilib_conversion_gives_the_camera_pose_in_the_field_frame() {
    let camera = camera_looking_down_x([-1.0, 0.3, 1.2]);
    let flu = pose_array(&wpilib_camera_from_optical(camera));
    assert_close(flu[0], -1.0, 1e-9, "x");
    assert_close(flu[1], 0.3, 1e-9, "y");
    assert_close(flu[2], 1.2, 1e-9, "z");
    // Forward (+x FLU) is field +x: the quaternion is the identity.
    assert_close(flu[3].abs(), 1.0, 1e-9, "qw");
}

#[test]
fn multi_tag_pose_recovers_the_camera_in_the_field_frame() {
    let cal = calibration();
    let side = 0.2;
    let camera = camera_looking_down_x([-1.0, 0.3, 1.2]);
    let tags = [(3u32, facing_back_tag(5.0, 0.0, 1.0)), (7u32, facing_back_tag(5.0, 1.0, 1.0))];
    let known = KnownTagPoses::new(
        tags.iter()
            .map(|(id, pose)| KnownTag {
                id: *id,
                side,
                reference_from_tag: eidos_reference_from_tag(*pose),
            })
            .collect(),
    );
    let observations: Vec<TagCorners> = tags
        .iter()
        .map(|(id, pose)| TagCorners {
            id: *id,
            corners: project(&camera, &eidos_reference_from_tag(*pose), side, &cal),
        })
        .collect();

    let mut solver = MultiTagPoseSolver::new(known, MultiTagPoseConfig::default());
    let mut out = MultiTagPose::default();
    assert!(solver.solve_into(&observations, &cal, None, &mut out));
    assert!(out.valid);
    assert_eq!(out.inlier_tags, 2);
    assert_pose_close(&out.reference_from_camera, &camera, 1e-3);

    let flu = pose_array(&wpilib_camera_from_optical(out.reference_from_camera));
    assert_close(flu[0], -1.0, 1e-3, "field x");
    assert_close(flu[1], 0.3, 1e-3, "field y");
}

#[test]
fn tag_frame_matches_the_wpilib_convention() {
    // A tag facing the origin from the field's +x, with its own +y pointing to the viewer's right
    // (the camera's image right): an Eidos corner at image right maps to WPILib +y.
    let eidos = eidos_reference_from_tag(facing_back_tag(5.0, 0.0, 1.0));
    // Eidos corner 1 (top-right of the marker, x = +h, y = -h, z = 0) in the field.
    let corner = square_object_corners(1.0)[1];
    let field = eidos_calibration::transform_point(&eidos, corner);
    assert_close(field.x, 5.0, 1e-9, "the marker plane is at the tag x");
    assert_close(field.y, -0.5, 1e-9, "image-right corner is field -y");
    assert_close(field.z, 1.0 + 0.5, 1e-9, "top corner is up");
}

#[test]
fn distortion_model_is_carried_through() {
    let cal = CameraCalibration {
        intrinsics: CameraIntrinsics { fx: 1.0, fy: 1.0, cx: 0.0, cy: 0.0, skew: 0.0 },
        distortion: DistortionCoefficients::default(),
        undistort_iters: 5,
        lens_model: LensModel::Fisheye,
    };
    assert_eq!(cal.lens_model, LensModel::Fisheye);
}

#[test]
fn the_3d_box_rises_from_the_tag_towards_the_camera() {
    // A 0.2 m tag 2 m straight ahead, facing the camera (Eidos tag frame = camera axes).
    let cal = calibration();
    let camera_from_tag = eidos_calibration::Se3Transform {
        rotation: eidos_calibration::RotationMatrix3 { m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] },
        translation: eidos_calibration::Translation3 { x: 0.0, y: 0.0, z: 2.0 },
    };
    let top = crate::detect::box_top(&camera_from_tag, 0.2, &cal).expect("in front of the camera");
    // The far face is 1.8 m away: corner 0 (top-left) at -0.1 / 1.8 * 800 from the centre.
    assert_close(top[0], 640.0 - 0.1 / 1.8 * 800.0, 1e-9, "u");
    assert_close(top[1], 400.0 - 0.1 / 1.8 * 800.0, 1e-9, "v");
    assert_close(top[4], 640.0 + 0.1 / 1.8 * 800.0, 1e-9, "u2");
}
