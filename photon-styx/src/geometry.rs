//! Pose conventions between Eidos, PhotonVision (WPILib) and OpenCV-style camera frames.
//!
//! - Eidos camera frames are optical: `x` right, `y` down, `z` forward (what OpenCV's `tvec` and
//!   `rvec` use). PhotonVision's per-tag poses are OpenCV-style too, converted in Java the same
//!   way as today, so single-tag results leave this crate in the optical frame.
//! - Field and robot poses are WPILib: field frame NWU, the camera's forward-left-up (FLU) frame.
//! - A WPILib AprilTag frame has `x` out of the tag's face, `y` to the viewer's right (the tag's
//!   own left-right reading), `z` up. An Eidos tag frame (`eidos_calibration::square_object_corners`)
//!   has `x` right, `y` down and `z` into the marker.

use eidos_calibration::{
    RotationMatrix3, Se3Transform, Translation3, flu_from_optical, quaternion_from_rotation_matrix,
    se3_compose, se3_identity, se3_inverse,
};

/// `eidos_tag_from_wpilib_tag`: a point in WPILib tag coordinates, times this matrix, is the same
/// point in Eidos tag coordinates. Columns are the WPILib axes in Eidos coordinates:
/// `x_w = -z_e`, `y_w = x_e`, `z_w = -y_e`.
const EIDOS_FROM_WPILIB_TAG: [[f64; 3]; 3] = [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]];

/// Rotation matrix product `a * b`.
pub fn mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

pub fn transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[j][i];
        }
    }
    out
}

/// A WPILib pose (`x, y, z` metres, unit quaternion `w, x, y, z`) as a rigid transform.
pub fn se3_from_wpilib(x: f64, y: f64, z: f64, qw: f64, qx: f64, qy: f64, qz: f64) -> Se3Transform {
    let n = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
    let (w, x_, y_, z_) = (qw / n, qx / n, qy / n, qz / n);
    let m = [
        [
            1.0 - 2.0 * (y_ * y_ + z_ * z_),
            2.0 * (x_ * y_ - z_ * w),
            2.0 * (x_ * z_ + y_ * w),
        ],
        [
            2.0 * (x_ * y_ + z_ * w),
            1.0 - 2.0 * (x_ * x_ + z_ * z_),
            2.0 * (y_ * z_ - x_ * w),
        ],
        [
            2.0 * (x_ * z_ - y_ * w),
            2.0 * (y_ * z_ + x_ * w),
            1.0 - 2.0 * (x_ * x_ + y_ * y_),
        ],
    ];
    Se3Transform {
        rotation: RotationMatrix3 { m },
        translation: Translation3 { x, y, z },
    }
}

/// The Eidos reference-from-tag transform of a tag whose WPILib field pose is `field_from_tag`.
/// The reference frame is the field frame; tag corners come from `square_object_corners`.
pub fn eidos_reference_from_tag(field_from_tag_wpilib: Se3Transform) -> Se3Transform {
    // A point p_e in Eidos tag coordinates is p_w = M^T p_e in WPILib tag coordinates.
    let wpilib_from_eidos = transpose(EIDOS_FROM_WPILIB_TAG);
    Se3Transform {
        rotation: RotationMatrix3 {
            m: mat_mul(field_from_tag_wpilib.rotation.m, wpilib_from_eidos),
        },
        translation: field_from_tag_wpilib.translation,
    }
}

/// `reference_from_camera` in the optical frame, re-expressed in the camera's forward-left-up
/// frame (what WPILib calls the robot-to-camera convention).
pub fn wpilib_camera_from_optical(reference_from_optical: Se3Transform) -> Se3Transform {
    se3_compose(reference_from_optical, se3_inverse(flu_from_optical()))
}

/// Pose as `[x, y, z, qw, qx, qy, qz]`.
pub fn pose_array(pose: &Se3Transform) -> [f64; 7] {
    let q = quaternion_from_rotation_matrix(pose.rotation);
    [
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        q.w,
        q.x,
        q.y,
        q.z,
    ]
}

pub fn identity() -> Se3Transform {
    se3_identity()
}
