//! Calibration boards on the Styx backend: chessboard and ChArUco detection in a Styx frame, and
//! the camera solve over the accumulated views. PhotonVision's calibration flow keeps its UI, its
//! snapshot flow and its saved coefficients; these functions replace OpenCV's
//! `findChessboardCorners`, `CharucoDetector` and `calibrateCameraExtended` with Eidos's
//! `calibration_target` detectors and `calibrate_camera` (Zhang's initialisation and bundle
//! adjustment, pinhole or fisheye).

use eidos_aruco::ArucoDictionaryKind;
use eidos_calibration::{
    CalibrationPointPair, CalibrationView, CameraCalibrationConfig, LensModel, PixelCoordinate,
    Point2, calibrate_camera,
};
use eidos_core::FrameLease;
use eidos_detectors::calibration_target::{
    CharucoBoard, CharucoDetector, CharucoLayout, ChessboardBoard, ChessboardDetector,
};
use serde::Deserialize;

/// PhotonVision refuses a ChArUco snapshot with fewer corners (its `hasValidCharucoDetections`).
pub const MIN_CHARUCO_CORNERS: usize = 10;

/// A board as PhotonVision's calibration UI describes it (`photon-core` `Calibrate3dPipeline`).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoardConfig {
    /// `"chessboard"` or `"charuco"`.
    pub kind: String,
    /// The board's squares across and down (PhotonVision's board width and height).
    pub squares_x: u32,
    pub squares_y: u32,
    /// Square edge, metres.
    pub square_m: f64,
    /// Marker edge, metres (ChArUco only).
    #[serde(default)]
    pub marker_m: f64,
    /// PhotonVision's dictionary name, e.g. `"4x4_1000"` (ChArUco only).
    #[serde(default = "default_dictionary")]
    pub dictionary: String,
    /// OpenCV's legacy ChArUco pattern (ChArUco only).
    #[serde(default)]
    pub legacy: bool,
}

fn default_dictionary() -> String {
    "4x4_1000".into()
}

/// A validated board. Corner ids are the detector's: row-major over the inner corners of a
/// chessboard, and the ChArUco chessboard corners of a ChArUco board.
#[derive(Debug, Clone)]
pub enum Board {
    Chessboard(ChessboardBoard),
    Charuco(CharucoBoard),
}

impl BoardConfig {
    /// Parse and validate a board description (JSON).
    pub fn parse(json: &str) -> Result<Board, String> {
        let config: Self = serde_json::from_str(json).map_err(|e| format!("bad board: {e}"))?;
        config.board()
    }

    pub fn board(&self) -> Result<Board, String> {
        match self.kind.as_str() {
            "chessboard" => {
                if self.squares_x < 3 || self.squares_y < 3 {
                    return Err(format!(
                        "a chessboard needs at least 3 squares per side, got {} x {}",
                        self.squares_x, self.squares_y
                    ));
                }
                // PhotonVision counts squares; the detector counts the inner corners between them.
                ChessboardBoard::new(self.squares_y - 1, self.squares_x - 1, self.square_m)
                    .map(Board::Chessboard)
                    .map_err(|e| e.to_string())
            }
            "charuco" => {
                let dictionary = dictionary(&self.dictionary)?;
                let layout = if self.legacy {
                    CharucoLayout::Legacy
                } else {
                    CharucoLayout::Standard
                };
                CharucoBoard::new(
                    self.squares_x,
                    self.squares_y,
                    self.square_m,
                    self.marker_m,
                    dictionary,
                )
                .map(|board| board.with_layout(layout))
                .map(Board::Charuco)
                .map_err(|e| e.to_string())
            }
            other => Err(format!(
                "unknown board kind '{other}' (chessboard or charuco)"
            )),
        }
    }
}

/// PhotonVision's dictionary names (`UICalibrationData.TagFamily`) that Eidos has.
pub fn dictionary(name: &str) -> Result<ArucoDictionaryKind, String> {
    use ArucoDictionaryKind as K;
    Ok(match name {
        "4x4_1000" => K::Aruco4x4_1000,
        "5x5_1000" => K::Aruco5x5_1000,
        "6x6_1000" => K::Aruco6x6_1000,
        "7x7_1000" => K::Aruco7x7_1000,
        other => {
            return Err(format!(
                "dictionary '{other}' is not supported on the Styx backend (4x4_1000, 5x5_1000, 6x6_1000, 7x7_1000)"
            ));
        }
    })
}

/// PhotonVision's lens models that the Styx backend solves: the OpenCV pinhole model and the
/// OpenCV fisheye model. The mrcal models have no Eidos solver.
pub fn lens_model(name: &str) -> Result<LensModel, String> {
    match name {
        "pinhole" => Ok(LensModel::Pinhole),
        "fisheye" => Ok(LensModel::Fisheye),
        other => Err(format!(
            "lens model '{other}' is not supported on the Styx backend (pinhole or fisheye)"
        )),
    }
}

impl Board {
    /// Board-plane point (metres, z = 0) of corner `id`.
    pub fn object_point(&self, id: u32) -> Option<Point2> {
        match self {
            Board::Chessboard(board) => board.object_point(id).map(|p| Point2 { x: p.x, y: p.y }),
            Board::Charuco(board) => board.object_point(id),
        }
    }
}

/// The corners of one board in one frame: their ids and their pixels (same order).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BoardDetection {
    pub ids: Vec<u32>,
    pub corners: Vec<PixelCoordinate>,
}

/// Finds the board in a frame, at the frame's full resolution. `Ok(None)` when the board is not
/// seen: a chessboard needs every inner corner, a ChArUco board at least [`MIN_CHARUCO_CORNERS`].
pub fn detect(board: &Board, frame: &FrameLease) -> Result<Option<BoardDetection>, String> {
    match board {
        Board::Chessboard(board) => {
            let found = ChessboardDetector::new(*board)
                .detect_frame(frame)
                .map_err(|e| format!("chessboard detection failed: {e:?}"))?;
            Ok(found
                .filter(|d| d.corners.len() == board.corner_count())
                .map(|d| BoardDetection {
                    ids: (0..d.corners.len() as u32).collect(),
                    corners: d.corners,
                }))
        }
        Board::Charuco(board) => {
            let mut detector = CharucoDetector::new(board.clone());
            let found = detector
                .detect(frame)
                .map_err(|e| format!("ChArUco detection failed: {e:?}"))?;
            Ok(found
                .filter(|d| d.corners.len() >= MIN_CHARUCO_CORNERS)
                .map(|d| BoardDetection {
                    ids: d.corners.iter().map(|c| c.id).collect(),
                    corners: d.corners.iter().map(|c| c.position).collect(),
                }))
        }
    }
}

/// One accepted snapshot: corner ids and the pixels of those corners.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct View {
    pub ids: Vec<u32>,
    pub corners: Vec<PixelCoordinate>,
}

/// The solved camera. `distortion` is in OpenCV's order: `k1 k2 p1 p2 k3` for the pinhole model,
/// `k1 k2 k3 k4` for the fisheye model (what PhotonVision stores).
#[derive(Debug, Clone, PartialEq)]
pub struct Solve {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub skew: f64,
    pub distortion: Vec<f64>,
    /// Per-view RMS reprojection error, pixels (input order).
    pub view_rms: Vec<f64>,
    /// RMS over every corner of every view, pixels.
    pub rms: f64,
    pub converged: bool,
}

/// Solve the camera from the views of `board` (corner ids map to board points).
pub fn solve(board: &Board, lens: LensModel, views: &[View]) -> Result<Solve, String> {
    let mut calibration_views = Vec::with_capacity(views.len());
    for (index, view) in views.iter().enumerate() {
        if view.ids.len() != view.corners.len() {
            return Err(format!(
                "view {index}: {} ids for {} corners",
                view.ids.len(),
                view.corners.len()
            ));
        }
        let mut points = Vec::with_capacity(view.ids.len());
        for (&id, &pixel) in view.ids.iter().zip(&view.corners) {
            let object = board
                .object_point(id)
                .ok_or_else(|| format!("view {index}: corner id {id} is not on the board"))?;
            points.push(CalibrationPointPair {
                object,
                image: Point2 {
                    x: pixel.u,
                    y: pixel.v,
                },
            });
        }
        calibration_views.push(CalibrationView::new(points));
    }
    let fit = calibrate_camera(
        &calibration_views,
        CameraCalibrationConfig {
            lens_model: lens,
            ..CameraCalibrationConfig::default()
        },
    )
    .map_err(|e| format!("calibration failed: {e:?}"))?;

    let d = fit.calibration.distortion;
    let distortion = match lens {
        LensModel::Pinhole => vec![d.k1, d.k2, d.p1, d.p2, d.k3],
        LensModel::Fisheye => vec![d.k1, d.k2, d.k3, d.k4],
    };
    let i = fit.calibration.intrinsics;
    Ok(Solve {
        fx: i.fx,
        fy: i.fy,
        cx: i.cx,
        cy: i.cy,
        skew: i.skew,
        distortion,
        view_rms: fit.view_rms,
        rms: fit.rms,
        converged: fit.converged,
    })
}

#[cfg(test)]
#[path = "calibrate_tests.rs"]
mod tests;
