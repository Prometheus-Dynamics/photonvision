//! Calibration on synthetic boards: views projected through a known camera, and boards rendered
//! through it (lens distortion applied per pixel), detected with Eidos's calibration_target
//! detectors, then solved back to the camera that made them. The renderer follows Eidos's own
//! calibration tests (`eidos-detectors` `calibration_target/tests/render.rs`) without the
//! supersampling and noise.

use super::*;
use eidos_aruco::{BitGrid, dictionary as aruco_dictionary};
use eidos_calibration::{
    CameraCalibration, CameraIntrinsics, DistortionCoefficients, Point3, RodriguesVector,
    project_camera_point, rodrigues_to_matrix, undistort_iter,
};
use eidos_detectors::calibration_target::{CharucoBoard as Charuco, ChessboardBoard as Chessboard};
use eidos_test_fixtures::{FixtureSize, FrameFixture};

const W: usize = 640;
const H: usize = 480;

/// The camera every test renders and solves.
fn camera(lens: LensModel) -> CameraCalibration {
    let distortion = match lens {
        LensModel::Pinhole => DistortionCoefficients {
            k1: -0.22,
            k2: 0.06,
            p1: 0.0009,
            p2: -0.0006,
            k3: -0.01,
            ..DistortionCoefficients::default()
        },
        LensModel::Fisheye => DistortionCoefficients {
            k1: 0.04,
            k2: -0.01,
            k3: 0.002,
            k4: -0.0005,
            ..DistortionCoefficients::default()
        },
    };
    CameraCalibration {
        intrinsics: CameraIntrinsics {
            fx: 640.0,
            fy: 642.0,
            cx: 322.0,
            cy: 238.0,
            skew: 0.0,
        },
        distortion,
        undistort_iters: 5,
        lens_model: lens,
    }
}

/// Board-to-camera pose: rotation matrix and translation (metres).
#[derive(Clone, Copy)]
struct Pose {
    r: [[f64; 3]; 3],
    t: [f64; 3],
}

fn pose(rvec: [f64; 3], t: [f64; 3]) -> Pose {
    let m = rodrigues_to_matrix(RodriguesVector {
        x: rvec[0],
        y: rvec[1],
        z: rvec[2],
    });
    Pose { r: m.m, t }
}

/// Ten views spread over the image with large tilts (the poses Eidos's own calibration tests use,
/// for a board centred on the origin's offset): distortion is observable out to the corners.
fn poses() -> Vec<Pose> {
    let views = [
        ([0.10, -0.12, 0.03], [-0.13, -0.09, 0.50]),
        ([0.45, 0.10, -0.05], [-0.10, -0.05, 0.55]),
        ([-0.50, 0.25, 0.10], [-0.08, -0.07, 0.48]),
        ([0.30, -0.55, 0.20], [-0.12, -0.04, 0.60]),
        ([0.65, -0.30, -0.15], [-0.09, -0.10, 0.52]),
        ([-0.20, 0.60, 0.25], [-0.11, -0.03, 0.58]),
        ([0.05, 0.05, 0.60], [-0.14, -0.08, 0.50]),
        ([0.55, 0.55, -0.30], [-0.07, -0.06, 0.62]),
        ([-0.40, -0.45, 0.10], [-0.12, -0.05, 0.54]),
        ([0.20, 0.35, -0.45], [-0.10, -0.04, 0.62]),
    ];
    views.iter().map(|(r, t)| pose(*r, *t)).collect()
}

fn board_to_camera(p: &Pose, x: f64, y: f64) -> Point3 {
    Point3 {
        x: p.r[0][0] * x + p.r[0][1] * y + p.t[0],
        y: p.r[1][0] * x + p.r[1][1] * y + p.t[1],
        z: p.r[2][0] * x + p.r[2][1] * y + p.t[2],
    }
}

fn corner_count(board: &Board) -> u32 {
    match board {
        Board::Chessboard(b) => b.corner_count() as u32,
        Board::Charuco(b) => b.corner_count() as u32,
    }
}

fn chessboard(squares_x: u32, squares_y: u32) -> Board {
    BoardConfig {
        kind: "chessboard".into(),
        squares_x,
        squares_y,
        square_m: 0.03,
        marker_m: 0.0,
        dictionary: default_dictionary(),
        legacy: false,
    }
    .board()
    .expect("chessboard")
}

fn charuco(squares_x: u32, squares_y: u32) -> Board {
    BoardConfig {
        kind: "charuco".into(),
        squares_x,
        squares_y,
        square_m: 0.04,
        marker_m: 0.03,
        dictionary: "4x4_1000".into(),
        legacy: false,
    }
    .board()
    .expect("charuco")
}

/// Views projected exactly through `cam` (no image, no detector).
fn projected_views(board: &Board, cam: &CameraCalibration, poses: &[Pose]) -> Vec<View> {
    poses
        .iter()
        .map(|p| {
            let mut view = View::default();
            for id in 0..corner_count(board) {
                let object = board.object_point(id).expect("object point");
                if let Some(px) = project_camera_point(board_to_camera(p, object.x, object.y), *cam)
                {
                    view.ids.push(id);
                    view.corners.push(px);
                }
            }
            view
        })
        .collect()
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Solves `A x = rhs` where `A` has the given columns (Cramer's rule).
fn solve3(cols: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<[f64; 3]> {
    let a = [
        [cols[0][0], cols[1][0], cols[2][0]],
        [cols[0][1], cols[1][1], cols[2][1]],
        [cols[0][2], cols[1][2], cols[2][2]],
    ];
    let det = det3(a);
    if det.abs() < 1e-12 {
        return None;
    }
    let mut x = [0.0; 3];
    for (i, value) in x.iter_mut().enumerate() {
        let mut m = a;
        for (row, r) in m.iter_mut().enumerate() {
            r[i] = rhs[row];
        }
        *value = det3(m) / det;
    }
    Some(x)
}

/// A chessboard's luma: black where `(r + c)` is even, origin at the first inner corner.
fn chessboard_shade(board: Chessboard) -> impl Fn(f64, f64) -> u8 {
    let s = board.square_size();
    let cols = i64::from(board.inner_cols()) + 1;
    let rows = i64::from(board.inner_rows()) + 1;
    move |x: f64, y: f64| {
        let c = (x / s).floor() as i64 + 1;
        let r = (y / s).floor() as i64 + 1;
        if c < 0 || r < 0 || c >= cols || r >= rows {
            return 200;
        }
        if (r + c) % 2 == 0 { 35 } else { 225 }
    }
}

/// A ChArUco board's luma: black squares, and in the marker squares the board's marker bits.
fn charuco_shade(board: &Charuco) -> impl Fn(f64, f64) -> u8 + use<> {
    let s = board.square_size();
    let m = board.marker_size();
    let dict = aruco_dictionary(board.dictionary());
    let grid_size = usize::from(dict.marker_size);
    let bits: Vec<Option<BitGrid>> = (0..board.marker_count())
        .map(|index| {
            let id = board.marker_id(index)?;
            BitGrid::from_marker(dict, id, 0)
        })
        .collect();
    let (sx, sy) = (i64::from(board.squares_x()), i64::from(board.squares_y()));
    let mut index_of: Vec<Option<usize>> = vec![None; (sx * sy) as usize];
    let mut is_marker = vec![false; (sx * sy) as usize];
    let mut next = 0usize;
    for r in 0..sy {
        for c in 0..sx {
            let slot = (r * sx + c) as usize;
            if board.is_marker_square(r as u32, c as u32) {
                is_marker[slot] = true;
                index_of[slot] = Some(next);
                next += 1;
            }
        }
    }
    move |x: f64, y: f64| {
        let c = (x / s).floor() as i64 + 1;
        let r = (y / s).floor() as i64 + 1;
        if c < 0 || r < 0 || c >= sx || r >= sy {
            return 200;
        }
        let slot = (r * sx + c) as usize;
        if !is_marker[slot] {
            return 35;
        }
        let index = index_of[slot].unwrap_or(usize::MAX);
        let cx = (c as f64 - 0.5) * s;
        let cy = (r as f64 - 0.5) * s;
        let u = (x - (cx - m * 0.5)) / m;
        let v = (y - (cy - m * 0.5)) / m;
        if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
            return 225;
        }
        let cells = grid_size + 2;
        let ci = (u * cells as f64) as usize;
        let cj = (v * cells as f64) as usize;
        if ci == 0 || cj == 0 || ci == cells - 1 || cj == cells - 1 {
            return 35;
        }
        match bits.get(index).and_then(|g| g.as_ref()) {
            Some(grid) => {
                if grid.bit(ci - 1, cj - 1) {
                    240
                } else {
                    25
                }
            }
            None => 225,
        }
    }
}

/// The board as seen through `cam` from `pose`: each sub-sample's undistorted ray meets the board
/// plane, and the pixel is the mean of the board's shading at those points (2 x 2 per pixel, so
/// the checker edges are not aliased into the corners).
fn render(cam: &CameraCalibration, pose: &Pose, shade: &dyn Fn(f64, f64) -> u8) -> Vec<u8> {
    const OFFSETS: [f64; 2] = [-0.25, 0.25];
    let mut out = vec![200u8; W * H];
    let r = pose.r;
    let t = pose.t;
    let k = cam.intrinsics;
    for y in 0..H {
        for x in 0..W {
            let mut acc = 0u32;
            for dy in OFFSETS {
                for dx in OFFSETS {
                    let distorted = Point2 {
                        x: (x as f64 + dx - k.cx) / k.fx,
                        y: (y as f64 + dy - k.cy) / k.fy,
                    };
                    let ray = undistort_iter(distorted, cam.distortion, cam.lens_model, 20);
                    // lambda * (ray.x, ray.y, 1) = X * column0 + Y * column1 + t, for (X, Y, lambda).
                    let cols = [
                        [r[0][0], r[1][0], r[2][0]],
                        [r[0][1], r[1][1], r[2][1]],
                        [-ray.x, -ray.y, -1.0],
                    ];
                    let value = match solve3(cols, [-t[0], -t[1], -t[2]]) {
                        Some([bx, by, lambda]) if lambda > 0.0 => shade(bx, by),
                        _ => 200,
                    };
                    acc += u32::from(value);
                }
            }
            out[y * W + x] = (acc / 4) as u8;
        }
    }
    out
}

/// A frame of `luma` at the test size.
fn frame_of(luma: &[u8]) -> FrameLease {
    let mut frame = FrameFixture::new(FixtureSize {
        width: W as u32,
        height: H as u32,
    })
    .build()
    .expect("frame");
    frame
        .visible_rows_mut(0)
        .expect("rows")
        .for_each_row_mut(|y, mut row| {
            let data = row.data();
            for x in 0..W {
                data[x] = luma[y * W + x];
            }
        });
    frame
}

/// The camera a solve describes (for the mapping check).
fn as_camera(solved: &Solve, lens: LensModel) -> CameraCalibration {
    let mut d = DistortionCoefficients::default();
    match lens {
        LensModel::Pinhole => {
            d.k1 = solved.distortion[0];
            d.k2 = solved.distortion[1];
            d.p1 = solved.distortion[2];
            d.p2 = solved.distortion[3];
            d.k3 = solved.distortion[4];
        }
        LensModel::Fisheye => {
            d.k1 = solved.distortion[0];
            d.k2 = solved.distortion[1];
            d.k3 = solved.distortion[2];
            d.k4 = solved.distortion[3];
        }
    }
    CameraCalibration {
        intrinsics: CameraIntrinsics {
            fx: solved.fx,
            fy: solved.fy,
            cx: solved.cx,
            cy: solved.cy,
            skew: solved.skew,
        },
        distortion: d,
        undistort_iters: 5,
        lens_model: lens,
    }
}

/// The bounding box of every observed corner: the region the views cover.
fn coverage(views: &[View]) -> (f64, f64, f64, f64) {
    let mut bounds = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for corner in views.iter().flat_map(|v| v.corners.iter()) {
        bounds.0 = bounds.0.min(corner.u);
        bounds.1 = bounds.1.min(corner.v);
        bounds.2 = bounds.2.max(corner.u);
        bounds.3 = bounds.3.max(corner.v);
    }
    bounds
}

/// The worst pixel error of the solved camera against the truth over the covered region: each
/// pixel's ray through the truth, reprojected by the solve.
fn mapping_error(solved: &Solve, cam: &CameraCalibration, region: (f64, f64, f64, f64)) -> f64 {
    let got = as_camera(solved, cam.lens_model);
    let (u0, v0, u1, v1) = region;
    let mut worst = 0.0f64;
    for i in 0..=12 {
        for j in 0..=12 {
            let x = u0 + (u1 - u0) * f64::from(i) / 12.0;
            let y = v0 + (v1 - v0) * f64::from(j) / 12.0;
            let p = Point2 {
                x: (x - cam.intrinsics.cx) / cam.intrinsics.fx,
                y: (y - cam.intrinsics.cy) / cam.intrinsics.fy,
            };
            let ray = undistort_iter(p, cam.distortion, cam.lens_model, 20);
            let lifted = Point3 {
                x: ray.x,
                y: ray.y,
                z: 1.0,
            };
            if let (Some(want), Some(have)) = (
                project_camera_point(lifted, *cam),
                project_camera_point(lifted, got),
            ) {
                worst = worst.max(((want.u - have.u).powi(2) + (want.v - have.v).powi(2)).sqrt());
            }
        }
    }
    worst
}

/// What a solve must reproduce, against the camera that made the views.
struct Tolerance {
    /// Focal lengths, as a fraction of the truth.
    focal: f64,
    /// Principal point, pixels.
    point_px: f64,
    /// Worst pixel error of the mapping (undistorted rays reprojected) over the covered region.
    mapping_px: f64,
    /// Each distortion term, absolute (`None`: not checked).
    distortion: Option<f64>,
}

/// Exact views: the solver itself, held to tight bounds (Eidos's own calibration bounds, with a
/// tighter mapping error).
const EXACT: Tolerance = Tolerance {
    focal: 0.01,
    point_px: 3.0,
    mapping_px: 1.0,
    distortion: Some(0.03),
};

/// Views rendered and detected: the detectors' corners carry sub-pixel bias, which the renders
/// (2 x 2 supersampled, 8-bit) do not hide. The bounds are what the calibration has to hold to
/// for PhotonVision (focal lengths within 1 %, undistortion within 2.5 px over the boards).
const RENDERED: Tolerance = Tolerance {
    focal: 0.01,
    point_px: 3.0,
    mapping_px: 2.5,
    distortion: None,
};

fn assert_recovers(
    solved: &Solve,
    cam: &CameraCalibration,
    region: (f64, f64, f64, f64),
    tol: Tolerance,
) {
    let k = cam.intrinsics;
    let focal = |got: f64, want: f64| (got - want).abs() / want;
    assert!(solved.converged, "the solve did not converge");
    assert!(
        focal(solved.fx, k.fx) < tol.focal,
        "fx {} vs {}",
        solved.fx,
        k.fx
    );
    assert!(
        focal(solved.fy, k.fy) < tol.focal,
        "fy {} vs {}",
        solved.fy,
        k.fy
    );
    assert!(
        (solved.cx - k.cx).abs() < tol.point_px,
        "cx {} vs {}",
        solved.cx,
        k.cx
    );
    assert!(
        (solved.cy - k.cy).abs() < tol.point_px,
        "cy {} vs {}",
        solved.cy,
        k.cy
    );
    let mapping = mapping_error(solved, cam, region);
    assert!(mapping < tol.mapping_px, "mapping error {mapping:.3} px");
    if let Some(bound) = tol.distortion {
        let d = cam.distortion;
        let want: Vec<f64> = match cam.lens_model {
            LensModel::Pinhole => vec![d.k1, d.k2, d.p1, d.p2, d.k3],
            LensModel::Fisheye => vec![d.k1, d.k2, d.k3, d.k4],
        };
        assert_eq!(solved.distortion.len(), want.len());
        for (i, (got, want)) in solved.distortion.iter().zip(&want).enumerate() {
            assert!(
                (got - want).abs() < bound,
                "distortion term {i}: {got} vs {want}"
            );
        }
    }
    assert!(solved.rms < 0.3, "rms {} px", solved.rms);
}

#[test]
fn synthetic_views_recover_the_pinhole_camera() {
    let board = chessboard(9, 7);
    let cam = camera(LensModel::Pinhole);
    let views = projected_views(&board, &cam, &poses());
    let region = coverage(&views);
    let solved = solve(&board, LensModel::Pinhole, &views).expect("solves");
    assert_recovers(&solved, &cam, region, EXACT);
}

#[test]
fn synthetic_views_recover_the_fisheye_camera() {
    let board = chessboard(9, 7);
    let cam = camera(LensModel::Fisheye);
    let views = projected_views(&board, &cam, &poses());
    let region = coverage(&views);
    let solved = solve(&board, LensModel::Fisheye, &views).expect("solves");
    assert_recovers(&solved, &cam, region, EXACT);
}

#[test]
fn rendered_chessboard_detects_all_corners_and_calibrates() {
    let board = chessboard(9, 7);
    let Board::Chessboard(chess) = board.clone() else {
        panic!("chessboard");
    };
    let shade = chessboard_shade(chess);
    let cam = camera(LensModel::Pinhole);
    let mut views = Vec::new();
    for p in poses() {
        let luma = render(&cam, &p, &shade);
        let frame = frame_of(&luma);
        let found = detect(&board, &frame)
            .expect("detects")
            .expect("the whole chessboard is found");
        assert_eq!(found.corners.len(), chess.corner_count());
        views.push(View {
            ids: found.ids,
            corners: found.corners,
        });
    }
    let region = coverage(&views);
    let solved = solve(&board, LensModel::Pinhole, &views).expect("solves");
    assert_recovers(&solved, &cam, region, RENDERED);
}

#[test]
fn rendered_charuco_detects_corners_and_ids_and_calibrates() {
    let board = charuco(6, 5);
    let Board::Charuco(ref charuco_board) = board else {
        panic!("charuco");
    };
    let shade = charuco_shade(charuco_board);
    let cam = camera(LensModel::Pinhole);
    let mut views = Vec::new();
    for p in poses() {
        let luma = render(&cam, &p, &shade);
        let frame = frame_of(&luma);
        let found = detect(&board, &frame)
            .expect("detects")
            .expect("the ChArUco board is found");
        assert!(found.corners.len() >= MIN_CHARUCO_CORNERS);
        views.push(View {
            ids: found.ids,
            corners: found.corners,
        });
    }
    let region = coverage(&views);
    let solved = solve(&board, LensModel::Pinhole, &views).expect("solves");
    assert_recovers(&solved, &cam, region, RENDERED);
}

#[test]
fn a_blank_frame_has_no_board() {
    let blank = vec![200u8; W * H];
    let frame = frame_of(&blank);
    assert_eq!(detect(&chessboard(9, 7), &frame).expect("detects"), None);
    assert_eq!(detect(&charuco(6, 5), &frame).expect("detects"), None);
}

#[test]
fn board_descriptions_are_validated_with_clear_errors() {
    let too_small = BoardConfig {
        kind: "chessboard".into(),
        squares_x: 2,
        squares_y: 2,
        square_m: 0.03,
        marker_m: 0.0,
        dictionary: default_dictionary(),
        legacy: false,
    };
    assert!(
        too_small
            .board()
            .unwrap_err()
            .contains("at least 3 squares")
    );

    let unknown_kind = BoardConfig {
        kind: "dots".into(),
        ..too_small.clone()
    };
    assert!(
        unknown_kind
            .board()
            .unwrap_err()
            .contains("unknown board kind")
    );

    assert!(dictionary("3x3_50").unwrap_err().contains("not supported"));
    assert_eq!(
        dictionary("5x5_1000").unwrap(),
        ArucoDictionaryKind::Aruco5x5_1000
    );
    assert!(
        lens_model("opencv_12")
            .unwrap_err()
            .contains("not supported")
    );
    assert_eq!(lens_model("fisheye").unwrap(), LensModel::Fisheye);
    assert!(
        BoardConfig::parse("not json")
            .unwrap_err()
            .contains("bad board")
    );
}
