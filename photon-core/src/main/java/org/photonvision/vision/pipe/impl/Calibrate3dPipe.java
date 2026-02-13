/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.pipe.impl;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.LoadJNI.JNITypes;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.mrcal.MrCalJNI;
import org.photonvision.mrcal.MrCalJNI.MrCalResult;
import org.photonvision.vision.calibration.BoardObservation;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.CameraLensModel;
import org.photonvision.vision.calibration.JsonMatOfDouble;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.pipe.CVPipe;
import org.photonvision.vision.pipe.impl.FindBoardCornersPipe.FindBoardCornersPipeResult;

public class Calibrate3dPipe
        extends CVPipe<
                Calibrate3dPipe.CalibrationInput,
                CameraCalibrationCoefficients,
                Calibrate3dPipe.CalibratePipeParams> {
    private static final double FISHEYE_THETA_LIMIT_RAD = Math.toRadians(89.5);
    private static final double FISHEYE_MAX_DIAGONAL_FOV_DEG = 179.0;

    public static class CalibrationInput {
        final List<FindBoardCornersPipe.FindBoardCornersPipeResult> observations;
        final FrameStaticProperties imageProps;
        final Path imageSavePath;

        public CalibrationInput(
                List<FindBoardCornersPipeResult> observations,
                FrameStaticProperties imageProps,
                Path imageSavePath) {
            this.observations = observations;
            this.imageProps = imageProps;
            this.imageSavePath = imageSavePath;
        }
    }

    // For logging
    private static final Logger logger = new Logger(Calibrate3dPipe.class, LogGroup.General);

    // The Standard deviation of the estimated parameters
    private final Mat stdDeviationsIntrinsics = new Mat();
    private final Mat stdDeviationsExtrinsics = new Mat();

    // Contains the re projection error of each snapshot by re projecting the
    // corners we found and
    // finding the Euclidean distance between the actual corners.
    private final Mat perViewErrors = new Mat();

    /**
     * Runs the process for the pipe.
     *
     * @param in Input for pipe processing. In the format (Input image, object points, image points)
     * @return Result of processing.
     */
    @Override
    protected CameraCalibrationCoefficients process(CalibrationInput in) {
        var filteredIn =
                in.observations.stream()
                        .filter(
                                it ->
                                        it != null
                                                && it.imagePoints != null
                                                && it.objectPoints != null
                                                && it.size != null)
                        .toList();

        CameraCalibrationCoefficients ret;
        var start = System.nanoTime();

        boolean useFisheye = params.lensModel == CameraLensModel.LENSMODEL_OPENCV_FISHEYE;
        double fxGuess = in.imageProps.horizontalFocalLength;
        double fyGuess = in.imageProps.verticalFocalLength;
        if (useFisheye) {
            // For fisheye calibration, always seed from configured diagonal FOV instead of any
            // existing calibration intrinsics to avoid converging back to stale pinhole-like values.
            var fovs =
                    FrameStaticProperties.calculateHorizontalVerticalFoV(
                            in.imageProps.fov, in.imageProps.imageWidth, in.imageProps.imageHeight);
            double horizFovRad = Math.toRadians(fovs.getFirst());
            double vertFovRad = Math.toRadians(fovs.getSecond());
            fxGuess = (in.imageProps.imageWidth / 2.0) / Math.tan(horizFovRad / 2.0);
            fyGuess = (in.imageProps.imageHeight / 2.0) / Math.tan(vertFovRad / 2.0);
            logger.debug(
                    "Fisheye initial intrinsics guess from FOV="
                            + in.imageProps.fov
                            + "deg -> fx="
                            + fxGuess
                            + ", fy="
                            + fyGuess);
        }

        if (LoadJNI.hasLoaded(JNITypes.MRCAL) && params.useMrCal && !useFisheye) {
            logger.debug("Calibrating with mrcal!");
            ret =
                    calibrateMrcal(
                            filteredIn,
                            fxGuess,
                            fyGuess,
                            in.imageSavePath);
        } else {
            if (useFisheye) {
                logger.debug("Calibrating with opencv fisheye!");
            } else {
                logger.debug("Calibrating with opencv!");
            }
            ret =
                    calibrateOpenCV(
                            filteredIn,
                            fxGuess,
                            fyGuess,
                            in.imageSavePath);
        }
        var dt = System.nanoTime() - start;

        if (ret != null)
            logger.info(
                    "CALIBRATION SUCCESS for res "
                            + in.observations.get(0).size
                            + " in "
                            + dt / 1e6
                            + "ms! camMatrix: \n"
                            + Arrays.toString(ret.cameraIntrinsics.data)
                            + "\ndistortionCoeffs:\n"
                            + Arrays.toString(ret.distCoeffs.data)
                            + "\n");
        else logger.info("Calibration failed! Review log for more details");

        return ret;
    }

    protected CameraCalibrationCoefficients calibrateOpenCV(
            List<FindBoardCornersPipe.FindBoardCornersPipeResult> in,
            double fxGuess,
            double fyGuess,
            Path imageSavePath) {
        List<MatOfPoint3f> objPointsIn = in.stream().map(it -> it.objectPoints).toList();
        List<MatOfPoint2f> imgPointsIn = in.stream().map(it -> it.imagePoints).toList();
        List<MatOfFloat> levelsArr = in.stream().map(it -> it.levels).toList();

        if (objPointsIn.size() != imgPointsIn.size() || objPointsIn.size() != levelsArr.size()) {
            logger.error("objpts.size != imgpts.size");
            return null;
        }

        // And delete rows depending on the level -- otherwise, level has no impact for opencv
        List<MatOfPoint3f> objPoints = new ArrayList<>();
        List<MatOfPoint2f> imgPoints = new ArrayList<>();
        List<FindBoardCornersPipe.FindBoardCornersPipeResult> filteredSnapshots = new ArrayList<>();
        int minCornersForSolve =
                params.lensModel == CameraLensModel.LENSMODEL_OPENCV_FISHEYE ? 6 : 4;
        for (int i = 0; i < objPointsIn.size(); i++) {
            MatOfPoint3f objPtsOut = new MatOfPoint3f();
            MatOfPoint2f imgPtsOut = new MatOfPoint2f();

            deleteIgnoredPoints(
                    objPointsIn.get(i), imgPointsIn.get(i), levelsArr.get(i), objPtsOut, imgPtsOut);

            int keptPoints = objPtsOut.rows() * objPtsOut.cols();
            if (keptPoints < minCornersForSolve || keptPoints != (imgPtsOut.rows() * imgPtsOut.cols())) {
                objPtsOut.release();
                imgPtsOut.release();
                continue;
            }

            objPoints.add(objPtsOut);
            imgPoints.add(imgPtsOut);
            filteredSnapshots.add(in.get(i));
        }

        if (objPoints.size() < 3) {
            logger.error(
                    "Calibration aborted: only "
                            + objPoints.size()
                            + " valid snapshots with >= "
                            + minCornersForSolve
                            + " corners. Need at least 3.");
            return null;
        }

        if (params.lensModel == CameraLensModel.LENSMODEL_OPENCV_FISHEYE) {
            return calibrateOpenCVFisheye(
                    filteredSnapshots, objPoints, imgPoints, fxGuess, fyGuess, imageSavePath);
        }

        return calibrateOpenCVPinhole(
                filteredSnapshots, objPoints, imgPoints, fxGuess, fyGuess, imageSavePath);
    }

    protected CameraCalibrationCoefficients calibrateOpenCVPinhole(
            List<FindBoardCornersPipe.FindBoardCornersPipeResult> filteredSnapshots,
            List<MatOfPoint3f> objPoints,
            List<MatOfPoint2f> imgPoints,
            double fxGuess,
            double fyGuess,
            Path imageSavePath) {
        Mat cameraMatrix = new Mat(3, 3, CvType.CV_64F);
        MatOfDouble distortionCoefficients = new MatOfDouble();
        List<Mat> rvecs = new ArrayList<>();
        List<Mat> tvecs = new ArrayList<>();

        // initial camera matrix guess
        double cx = (filteredSnapshots.get(0).size.width / 2.0) - 0.5;
        double cy = (filteredSnapshots.get(0).size.height / 2.0) - 0.5;
        cameraMatrix.put(0, 0, new double[] {fxGuess, 0, cx, 0, fyGuess, cy, 0, 0, 1});

        try {
            // FindBoardCorners pipe outputs all the image points, object points, and frames
            // to calculate
            // imageSize from, other parameters are output Mats

            Calib3d.calibrateCameraExtended(
                    objPoints.stream().map(it -> (Mat) it).toList(),
                    imgPoints.stream().map(it -> (Mat) it).toList(),
                    new Size(filteredSnapshots.get(0).size.width, filteredSnapshots.get(0).size.height),
                    cameraMatrix,
                    distortionCoefficients,
                    rvecs,
                    tvecs,
                    stdDeviationsIntrinsics,
                    stdDeviationsExtrinsics,
                    perViewErrors,
                    Calib3d.CALIB_USE_LU + Calib3d.CALIB_USE_INTRINSIC_GUESS);
        } catch (Exception e) {
            logger.error("Calibration failed!", e);
            e.printStackTrace();
            return null;
        }

        JsonMatOfDouble cameraMatrixMat = JsonMatOfDouble.fromMat(cameraMatrix);
        JsonMatOfDouble distortionCoefficientsMat = JsonMatOfDouble.fromMat(distortionCoefficients);

        // Opencv is lame, so we can only assume all points are inliers
        var inliners =
                objPoints.stream()
                        .map(
                                it -> {
                                    var array = new boolean[it.rows() * it.cols()];
                                    Arrays.fill(array, true);
                                    return array;
                                })
                        .toList();

        var observations =
                createObservations(
                        filteredSnapshots,
                        cameraMatrix,
                        distortionCoefficients,
                        rvecs,
                        tvecs,
                        inliners,
                        new double[] {0, 0},
                        objPoints,
                        imgPoints,
                        imageSavePath,
                        CameraLensModel.LENSMODEL_OPENCV);

        if (observations.isEmpty()) {
            logger.error("Calibration aborted: no valid observations were produced.");
            cameraMatrix.release();
            distortionCoefficients.release();
            rvecs.forEach(Mat::release);
            tvecs.forEach(Mat::release);
            objPoints.forEach(Mat::release);
            imgPoints.forEach(Mat::release);
            return null;
        }

        cameraMatrix.release();
        distortionCoefficients.release();
        rvecs.forEach(Mat::release);
        tvecs.forEach(Mat::release);
        objPoints.forEach(Mat::release);
        imgPoints.forEach(Mat::release);

        return new CameraCalibrationCoefficients(
                filteredSnapshots.get(0).size,
                cameraMatrixMat,
                distortionCoefficientsMat,
                new double[0],
                observations,
                new Size(params.boardWidth, params.boardHeight),
                params.squareSize,
                CameraLensModel.LENSMODEL_OPENCV);
    }

    protected CameraCalibrationCoefficients calibrateOpenCVFisheye(
            List<FindBoardCornersPipe.FindBoardCornersPipeResult> in,
            List<MatOfPoint3f> objPoints,
            List<MatOfPoint2f> imgPoints,
            double fxGuess,
            double fyGuess,
            Path imageSavePath) {
        Mat cameraMatrix = Mat.eye(3, 3, CvType.CV_64F);
        Mat distortionCoefficients = Mat.zeros(4, 1, CvType.CV_64F);
        List<Mat> rvecs = new ArrayList<>();
        List<Mat> tvecs = new ArrayList<>();

        double cx = (in.get(0).size.width / 2.0) - 0.5;
        double cy = (in.get(0).size.height / 2.0) - 0.5;
        cameraMatrix.put(0, 0, fxGuess);
        cameraMatrix.put(1, 1, fyGuess);
        cameraMatrix.put(0, 2, cx);
        cameraMatrix.put(1, 2, cy);

        // OpenCV fisheye calibration can fail during InitExtrinsics when the first correspondence in
        // a snapshot normalizes to exactly zero. Reorder points so high-radius points come first while
        // preserving object/image point pairing.
        reorderPointsForFisheyeInitialization(objPoints, imgPoints, cx, cy);

        int flags = Calib3d.fisheye_CALIB_USE_INTRINSIC_GUESS | Calib3d.fisheye_CALIB_RECOMPUTE_EXTRINSIC;
        TermCriteria criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 100, 1e-6);

        var activeSnapshotIndices = new ArrayList<Integer>();
        for (int i = 0; i < objPoints.size(); i++) {
            activeSnapshotIndices.add(i);
        }

        boolean fisheyeSolved = false;
        while (activeSnapshotIndices.size() >= 3) {
            var attemptObjPoints = new ArrayList<Mat>();
            var attemptImgPoints = new ArrayList<Mat>();
            for (int idx : activeSnapshotIndices) {
                attemptObjPoints.add(objPoints.get(idx));
                attemptImgPoints.add(imgPoints.get(idx));
            }

            boolean solvedThisRound = false;
            Exception lastError = null;
            int[] flagOptions = new int[] {flags, Calib3d.fisheye_CALIB_RECOMPUTE_EXTRINSIC};
            double[][] scaleOptions =
                    new double[][] {
                        {1.0, 1.0},
                        {0.8, 0.8},
                        {1.2, 1.2},
                        {0.9, 1.1},
                        {1.1, 0.9}
                    };

            for (int attemptFlags : flagOptions) {
                for (double[] scale : scaleOptions) {
                    clearMatList(rvecs);
                    clearMatList(tvecs);
                    distortionCoefficients.setTo(new Scalar(0));

                    double seedFx = fxGuess * scale[0];
                    double seedFy = fyGuess * scale[1];
                    cameraMatrix.put(0, 0, seedFx);
                    cameraMatrix.put(1, 1, seedFy);
                    cameraMatrix.put(0, 2, cx);
                    cameraMatrix.put(1, 2, cy);

                    try {
                        Calib3d.fisheye_calibrate(
                                attemptObjPoints,
                                attemptImgPoints,
                                new Size(in.get(0).size.width, in.get(0).size.height),
                                cameraMatrix,
                                distortionCoefficients,
                                rvecs,
                                tvecs,
                                attemptFlags,
                                criteria);

                        if (isDegenerateFisheyeSolution(
                                cameraMatrix,
                                distortionCoefficients,
                                seedFx,
                                seedFy,
                                in.get(0).size.width,
                                in.get(0).size.height)) {
                            logger.warn(
                                    "fisheye_calibrate produced degenerate solution (unchanged intrinsics/zero distortion), retrying with alternate initialization.");
                            continue;
                        }

                        fisheyeSolved = true;
                        solvedThisRound = true;
                        break;
                    } catch (Exception e) {
                        lastError = e;
                    }
                }
                if (solvedThisRound) {
                    break;
                }
            }

            if (solvedThisRound) {
                break;
            }

            if (activeSnapshotIndices.size() <= 3) {
                if (lastError != null) {
                    logger.error("Fisheye calibration failed after retries!", lastError);
                } else {
                    logger.error(
                            "Fisheye calibration failed after retries: only degenerate solutions were produced.");
                }
                break;
            }

            int dropPos = findWorstSnapshotPositionForFisheye(activeSnapshotIndices, imgPoints);
            int droppedSnapshotIdx = activeSnapshotIndices.remove(dropPos);
            logger.warn(
                    "fisheye_calibrate failed to produce valid non-degenerate solution; dropping snapshot "
                            + droppedSnapshotIdx
                            + " and retrying with "
                            + activeSnapshotIndices.size()
                            + " snapshots.");
            if (lastError != null) {
                logger.debug(
                        "Last fisheye error before drop: "
                                + lastError.getClass().getSimpleName()
                                + ": "
                                + lastError.getMessage());
            }
        }

        if (!fisheyeSolved) {
            logger.error("Fisheye calibration failed to initialize extrinsics.");
            cameraMatrix.release();
            distortionCoefficients.release();
            clearMatList(rvecs);
            clearMatList(tvecs);
            objPoints.forEach(Mat::release);
            imgPoints.forEach(Mat::release);
            return null;
        }

        JsonMatOfDouble cameraMatrixMat = JsonMatOfDouble.fromMat(cameraMatrix);
        JsonMatOfDouble distortionCoefficientsMat = JsonMatOfDouble.fromMat(distortionCoefficients);

        var snapshotsForObs = new ArrayList<FindBoardCornersPipe.FindBoardCornersPipeResult>();
        var objPointsForObs = new ArrayList<MatOfPoint3f>();
        var imgPointsForObs = new ArrayList<MatOfPoint2f>();
        for (int idx : activeSnapshotIndices) {
            snapshotsForObs.add(in.get(idx));
            objPointsForObs.add(objPoints.get(idx));
            imgPointsForObs.add(imgPoints.get(idx));
        }
        List<Mat> rvecsForObs = rvecs;
        List<Mat> tvecsForObs = tvecs;

        // Always recover per-view poses from undistorted points.
        // This avoids malformed fisheye rvec/tvec outputs causing projectPoints assertions.
        {
            logger.debug(
                    "Recovering per-view fisheye extrinsics with solvePnP (rvecs="
                            + rvecs.size()
                            + ", tvecs="
                            + tvecs.size()
                            + ", snapshots="
                            + objPointsForObs.size()
                            + ").");

            var solvedSnapshots = new ArrayList<FindBoardCornersPipe.FindBoardCornersPipeResult>();
            var solvedObjPoints = new ArrayList<MatOfPoint3f>();
            var solvedImgPoints = new ArrayList<MatOfPoint2f>();
            var solvedRvecs = new ArrayList<Mat>();
            var solvedTvecs = new ArrayList<Mat>();

            for (int i = 0; i < objPointsForObs.size(); i++) {
                var undistorted = new MatOfPoint2f();
                var identity = Mat.eye(3, 3, CvType.CV_64F);
                var zeroDistortion = new MatOfDouble(0, 0, 0, 0);
                var rvec = new Mat();
                var tvec = new Mat();

                try {
                    Calib3d.fisheye_undistortPoints(
                            imgPointsForObs.get(i),
                            undistorted,
                            cameraMatrix,
                            distortionCoefficients,
                            identity,
                            cameraMatrix);

                    boolean solved =
                            Calib3d.solvePnP(
                                    objPointsForObs.get(i),
                                    undistorted,
                                    cameraMatrix,
                                    zeroDistortion,
                                    rvec,
                                    tvec);

                    if (!solved || rvec.empty() || tvec.empty()) {
                        rvec.release();
                        tvec.release();
                        continue;
                    }

                    solvedSnapshots.add(snapshotsForObs.get(i));
                    solvedObjPoints.add(objPointsForObs.get(i));
                    solvedImgPoints.add(imgPointsForObs.get(i));
                    solvedRvecs.add(rvec);
                    solvedTvecs.add(tvec);
                } catch (Exception e) {
                    logger.debug(
                            "Failed to recover fisheye extrinsics for snapshot "
                                    + i
                                    + ": "
                                    + e.getClass().getSimpleName()
                                    + " "
                                    + e.getMessage());
                    rvec.release();
                    tvec.release();
                } finally {
                    zeroDistortion.release();
                    identity.release();
                    undistorted.release();
                }
            }

            if (solvedSnapshots.size() < 3) {
                logger.error(
                        "Only "
                                + solvedSnapshots.size()
                                + " snapshots produced valid fisheye extrinsics.");
                cameraMatrix.release();
                distortionCoefficients.release();
                rvecs.forEach(Mat::release);
                tvecs.forEach(Mat::release);
                objPoints.forEach(Mat::release);
                imgPoints.forEach(Mat::release);
                solvedRvecs.forEach(Mat::release);
                solvedTvecs.forEach(Mat::release);
                return null;
            }

            rvecsForObs = solvedRvecs;
            tvecsForObs = solvedTvecs;
            snapshotsForObs = solvedSnapshots;
            objPointsForObs = solvedObjPoints;
            imgPointsForObs = solvedImgPoints;
        }

        var inliners =
                objPointsForObs.stream()
                        .map(
                                it -> {
                                    var array = new boolean[it.rows() * it.cols()];
                                    Arrays.fill(array, true);
                                    return array;
                                })
                        .toList();

        var observations =
                createObservations(
                        snapshotsForObs,
                        cameraMatrix,
                        distortionCoefficients,
                        rvecsForObs,
                        tvecsForObs,
                        inliners,
                        new double[] {0, 0},
                        objPointsForObs,
                        imgPointsForObs,
                        imageSavePath,
                        CameraLensModel.LENSMODEL_OPENCV_FISHEYE);

        if (observations.isEmpty()) {
            logger.error("Calibration aborted in fisheye observation pass: no valid observations were produced.");
            cameraMatrix.release();
            distortionCoefficients.release();
            rvecs.forEach(Mat::release);
            tvecs.forEach(Mat::release);
            if (rvecsForObs != rvecs) {
                rvecsForObs.forEach(Mat::release);
            }
            if (tvecsForObs != tvecs) {
                tvecsForObs.forEach(Mat::release);
            }
            objPoints.forEach(Mat::release);
            imgPoints.forEach(Mat::release);
            return null;
        }

        cameraMatrix.release();
        distortionCoefficients.release();
        rvecs.forEach(Mat::release);
        tvecs.forEach(Mat::release);
        if (rvecsForObs != rvecs) {
            rvecsForObs.forEach(Mat::release);
        }
        if (tvecsForObs != tvecs) {
            tvecsForObs.forEach(Mat::release);
        }
        objPoints.forEach(Mat::release);
        imgPoints.forEach(Mat::release);

        return new CameraCalibrationCoefficients(
                snapshotsForObs.get(0).size,
                cameraMatrixMat,
                distortionCoefficientsMat,
                new double[0],
                observations,
                new Size(params.boardWidth, params.boardHeight),
                params.squareSize,
                CameraLensModel.LENSMODEL_OPENCV_FISHEYE);
    }

    private void clearMatList(List<Mat> mats) {
        mats.forEach(Mat::release);
        mats.clear();
    }

    private boolean isDegenerateFisheyeSolution(
            Mat cameraMatrix,
            Mat distortionCoefficients,
            double seedFx,
            double seedFy,
            double imageWidth,
            double imageHeight) {
        if (cameraMatrix == null
                || cameraMatrix.empty()
                || distortionCoefficients == null
                || distortionCoefficients.empty()) {
            return true;
        }

        double fx = cameraMatrix.get(0, 0)[0];
        double fy = cameraMatrix.get(1, 1)[0];
        double alpha = cameraMatrix.get(0, 1)[0];
        double cx = cameraMatrix.get(0, 2)[0];
        double cy = cameraMatrix.get(1, 2)[0];
        if (!Double.isFinite(fx)
                || !Double.isFinite(fy)
                || !Double.isFinite(alpha)
                || !Double.isFinite(cx)
                || !Double.isFinite(cy)
                || fx <= 0
                || fy <= 0) {
            return true;
        }

        double relFx = Math.abs(fx - seedFx) / Math.max(1.0, Math.abs(seedFx));
        double relFy = Math.abs(fy - seedFy) / Math.max(1.0, Math.abs(seedFy));

        if (Double.isFinite(seedFx) && seedFx > 0) {
            if (fx < seedFx * 0.4 || fx > seedFx * 2.5) {
                logger.warn(
                        "fisheye_calibrate solution rejected: fx drifted too far from seed (seed="
                                + seedFx
                                + ", solved="
                                + fx
                                + ").");
                return true;
            }
        }

        if (Double.isFinite(seedFy) && seedFy > 0) {
            if (fy < seedFy * 0.4 || fy > seedFy * 2.5) {
                logger.warn(
                        "fisheye_calibrate solution rejected: fy drifted too far from seed (seed="
                                + seedFy
                                + ", solved="
                                + fy
                                + ").");
                return true;
            }
        }

        if (cx < -0.5 * imageWidth
                || cx > 1.5 * imageWidth
                || cy < -0.5 * imageHeight
                || cy > 1.5 * imageHeight) {
            logger.warn(
                    "fisheye_calibrate solution rejected: principal point out of bounds (cx="
                            + cx
                            + ", cy="
                            + cy
                            + ").");
            return true;
        }

        double distMagnitude = 0.0;
        int distTotal = (int) (distortionCoefficients.total() * distortionCoefficients.channels());
        if (distTotal <= 0) {
            return true;
        }
        double[] distData = new double[distTotal];
        distortionCoefficients.get(0, 0, distData);
        for (double d : distData) {
            if (!Double.isFinite(d)) {
                return true;
            }
            distMagnitude += Math.abs(d);
        }

        if (relFx < 1e-6 && relFy < 1e-6 && distMagnitude < 1e-8) {
            return true;
        }

        if (!isFisheyeThetaMappingMonotonic(distData, FISHEYE_THETA_LIMIT_RAD)) {
            logger.warn("fisheye_calibrate solution rejected: non-monotonic theta distortion mapping.");
            return true;
        }

        double requiredThetaD =
                requiredDistortedThetaForImage(cameraMatrix, imageWidth, imageHeight);
        double supportedThetaD = maxSupportedDistortedTheta(distData, FISHEYE_THETA_LIMIT_RAD);
        if (!Double.isFinite(requiredThetaD)
                || !Double.isFinite(supportedThetaD)
                || requiredThetaD <= 0
                || supportedThetaD <= 0) {
            return true;
        }

        // Reject fisheye fits that cannot represent the image corners in distorted-theta space.
        // These solve "successfully" but collapse to unrealistically narrow FOV.
        if (supportedThetaD < requiredThetaD * 0.99) {
            logger.warn(
                    "fisheye_calibrate solution rejected: insufficient theta coverage (required="
                            + requiredThetaD
                            + ", supported="
                            + supportedThetaD
                            + ").");
            return true;
        }

        double solvedDiagFovDeg =
                fisheyeDiagonalFovDegrees(cameraMatrix, distData, imageWidth, imageHeight);
        if (!Double.isFinite(solvedDiagFovDeg)
                || solvedDiagFovDeg <= 0
                || solvedDiagFovDeg > FISHEYE_MAX_DIAGONAL_FOV_DEG) {
            logger.warn(
                    "fisheye_calibrate solution rejected: invalid diagonal FOV ("
                            + solvedDiagFovDeg
                            + " deg).");
            return true;
        }

        return false;
    }

    private double requiredDistortedThetaForImage(
            Mat cameraMatrix, double imageWidth, double imageHeight) {
        double fx = cameraMatrix.get(0, 0)[0];
        double alpha = cameraMatrix.get(0, 1)[0];
        double cx = cameraMatrix.get(0, 2)[0];
        double fy = cameraMatrix.get(1, 1)[0];
        double cy = cameraMatrix.get(1, 2)[0];

        if (!Double.isFinite(fx)
                || !Double.isFinite(fy)
                || !Double.isFinite(alpha)
                || !Double.isFinite(cx)
                || !Double.isFinite(cy)
                || fx == 0
                || fy == 0) {
            return Double.NaN;
        }

        double[][] corners =
                new double[][] {
                    {0, 0},
                    {Math.max(0.0, imageWidth - 1.0), 0},
                    {0, Math.max(0.0, imageHeight - 1.0)},
                    {Math.max(0.0, imageWidth - 1.0), Math.max(0.0, imageHeight - 1.0)}
                };

        double required = 0.0;
        for (double[] corner : corners) {
            double u = corner[0];
            double v = corner[1];
            double y = (v - cy) / fy;
            double x = (u - cx - alpha * y) / fx;
            if (!Double.isFinite(x) || !Double.isFinite(y)) {
                return Double.NaN;
            }
            required = Math.max(required, Math.hypot(x, y));
        }

        return required;
    }

    private double maxSupportedDistortedTheta(double[] distData, double thetaLimitRad) {
        if (distData.length == 0) {
            return 0.0;
        }

        final int samples = 4096;
        double max = 0.0;
        for (int i = 0; i <= samples; i++) {
            double theta = thetaLimitRad * i / samples;
            double distorted = fisheyeDistortedTheta(theta, distData);
            if (Double.isFinite(distorted)) {
                max = Math.max(max, distorted);
            }
        }
        return max;
    }

    private boolean isFisheyeThetaMappingMonotonic(double[] distData, double thetaLimitRad) {
        if (distData.length == 0 || !Double.isFinite(thetaLimitRad) || thetaLimitRad <= 0) {
            return false;
        }

        final int samples = 2048;
        double previous = 0.0;
        for (int i = 1; i <= samples; i++) {
            double theta = thetaLimitRad * i / samples;
            double distorted = fisheyeDistortedTheta(theta, distData);
            double derivative = fisheyeDistortedThetaDerivative(theta, distData);
            if (!Double.isFinite(distorted) || !Double.isFinite(derivative) || derivative <= 1e-6) {
                return false;
            }
            if (distorted <= previous) {
                return false;
            }
            previous = distorted;
        }
        return true;
    }

    private double fisheyeDistortedTheta(double theta, double[] distData) {
        double theta2 = theta * theta;
        double thetaPow = theta2;
        double sum = 1.0;
        for (double k : distData) {
            sum += k * thetaPow;
            thetaPow *= theta2;
        }
        return theta * sum;
    }

    private double fisheyeDistortedThetaDerivative(double theta, double[] distData) {
        double theta2 = theta * theta;
        double thetaPow = theta2;
        double derivative = 1.0;
        for (int i = 0; i < distData.length; i++) {
            derivative += (2.0 * i + 3.0) * distData[i] * thetaPow;
            thetaPow *= theta2;
        }
        return derivative;
    }

    private double fisheyeDiagonalFovDegrees(
            Mat cameraMatrix, double[] distData, double imageWidth, double imageHeight) {
        var tl = fisheyePixelToRay(cameraMatrix, distData, 0.0, 0.0);
        var tr = fisheyePixelToRay(cameraMatrix, distData, Math.max(0.0, imageWidth - 1.0), 0.0);
        var bl = fisheyePixelToRay(cameraMatrix, distData, 0.0, Math.max(0.0, imageHeight - 1.0));
        var br =
                fisheyePixelToRay(
                        cameraMatrix,
                        distData,
                        Math.max(0.0, imageWidth - 1.0),
                        Math.max(0.0, imageHeight - 1.0));

        if (tl == null || tr == null || bl == null || br == null) {
            return Double.NaN;
        }

        double diag1 = rayAngleDegrees(tl, br);
        double diag2 = rayAngleDegrees(tr, bl);
        return Math.max(diag1, diag2);
    }

    private double[] fisheyePixelToRay(
            Mat cameraMatrix, double[] distData, double u, double v) {
        double fx = cameraMatrix.get(0, 0)[0];
        double alpha = cameraMatrix.get(0, 1)[0];
        double cx = cameraMatrix.get(0, 2)[0];
        double fy = cameraMatrix.get(1, 1)[0];
        double cy = cameraMatrix.get(1, 2)[0];

        if (!Double.isFinite(fx)
                || !Double.isFinite(fy)
                || !Double.isFinite(alpha)
                || !Double.isFinite(cx)
                || !Double.isFinite(cy)
                || fx <= 0
                || fy <= 0) {
            return null;
        }

        double y = (v - cy) / fy;
        double x = (u - cx - alpha * y) / fx;
        if (!Double.isFinite(x) || !Double.isFinite(y)) {
            return null;
        }

        double distortedRadius = Math.hypot(x, y);
        if (distortedRadius < 1e-12) {
            return new double[] {0, 0, 1};
        }

        double theta = fisheyeThetaFromDistortedRadius(distortedRadius, distData);
        if (!Double.isFinite(theta) || theta < 0 || theta >= FISHEYE_THETA_LIMIT_RAD) {
            return null;
        }

        double r = Math.tan(theta);
        if (!Double.isFinite(r) || r < 0) {
            return null;
        }

        double scale = r / distortedRadius;
        double rx = x * scale;
        double ry = y * scale;
        double rz = 1.0;
        double norm = Math.sqrt(rx * rx + ry * ry + rz * rz);
        if (!Double.isFinite(norm) || norm <= 0) {
            return null;
        }

        return new double[] {rx / norm, ry / norm, rz / norm};
    }

    private double fisheyeThetaFromDistortedRadius(double distortedRadius, double[] distData) {
        if (!Double.isFinite(distortedRadius) || distortedRadius < 0) {
            return Double.NaN;
        }
        if (distortedRadius == 0) {
            return 0;
        }

        double maxDistorted = fisheyeDistortedTheta(FISHEYE_THETA_LIMIT_RAD, distData);
        if (!Double.isFinite(maxDistorted) || maxDistorted <= 0 || distortedRadius > maxDistorted) {
            return Double.NaN;
        }

        double lo = 0.0;
        double hi = FISHEYE_THETA_LIMIT_RAD;
        for (int i = 0; i < 80; i++) {
            double mid = (lo + hi) / 2.0;
            double value = fisheyeDistortedTheta(mid, distData);
            if (!Double.isFinite(value)) {
                return Double.NaN;
            }
            if (value < distortedRadius) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return (lo + hi) / 2.0;
    }

    private double rayAngleDegrees(double[] a, double[] b) {
        if (a == null || b == null || a.length != 3 || b.length != 3) {
            return Double.NaN;
        }

        double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        if (!Double.isFinite(dot)) {
            return Double.NaN;
        }

        dot = Math.max(-1.0, Math.min(1.0, dot));
        return Math.toDegrees(Math.acos(dot));
    }

    private void reorderPointsForFisheyeInitialization(
            List<MatOfPoint3f> objPoints, List<MatOfPoint2f> imgPoints, double cx, double cy) {
        int snapshotCount = Math.min(objPoints.size(), imgPoints.size());
        for (int i = 0; i < snapshotCount; i++) {
            Point3[] obj = objPoints.get(i).toArray();
            Point[] img = imgPoints.get(i).toArray();
            int n = Math.min(obj.length, img.length);
            if (n < 2) {
                continue;
            }

            Integer[] order = new Integer[n];
            for (int j = 0; j < n; j++) {
                order[j] = j;
            }
            Arrays.sort(
                    order,
                    (a, b) ->
                            Double.compare(
                                    radiusSq(img[b], cx, cy), radiusSq(img[a], cx, cy)));

            Point3[] reorderedObj = new Point3[n];
            Point[] reorderedImg = new Point[n];
            for (int j = 0; j < n; j++) {
                int idx = order[j];
                reorderedObj[j] = obj[idx];
                reorderedImg[j] = img[idx];
            }

            objPoints.get(i).fromArray(reorderedObj);
            imgPoints.get(i).fromArray(reorderedImg);
        }
    }

    private double radiusSq(Point p, double cx, double cy) {
        double dx = p.x - cx;
        double dy = p.y - cy;
        return dx * dx + dy * dy;
    }

    private int findWorstSnapshotPositionForFisheye(
            List<Integer> activeSnapshotIndices, List<MatOfPoint2f> imgPoints) {
        int worstPosition = 0;
        double worstScore = Double.POSITIVE_INFINITY;

        for (int pos = 0; pos < activeSnapshotIndices.size(); pos++) {
            int snapshotIdx = activeSnapshotIndices.get(pos);
            double score = fisheyeSnapshotQuality(imgPoints.get(snapshotIdx));
            if (score < worstScore) {
                worstScore = score;
                worstPosition = pos;
            }
        }

        return worstPosition;
    }

    private double fisheyeSnapshotQuality(MatOfPoint2f pointsMat) {
        Point[] points = pointsMat.toArray();
        if (points.length < 6) {
            return -1.0;
        }

        double minX = Double.POSITIVE_INFINITY;
        double minY = Double.POSITIVE_INFINITY;
        double maxX = Double.NEGATIVE_INFINITY;
        double maxY = Double.NEGATIVE_INFINITY;
        for (Point p : points) {
            if (!Double.isFinite(p.x) || !Double.isFinite(p.y)) {
                return -1.0;
            }
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x);
            maxY = Math.max(maxY, p.y);
        }
        double bboxArea = Math.max(0.0, (maxX - minX) * (maxY - minY));

        double minPairDist = Double.POSITIVE_INFINITY;
        for (int i = 0; i < points.length; i++) {
            for (int j = i + 1; j < points.length; j++) {
                double dx = points[i].x - points[j].x;
                double dy = points[i].y - points[j].y;
                double dist = Math.hypot(dx, dy);
                if (dist < minPairDist) {
                    minPairDist = dist;
                }
            }
        }

        if (!Double.isFinite(minPairDist)) {
            minPairDist = 0.0;
        }

        return bboxArea * Math.max(minPairDist, 1e-6);
    }

    protected CameraCalibrationCoefficients calibrateMrcal(
            List<FindBoardCornersPipe.FindBoardCornersPipeResult> in,
            double fxGuess,
            double fyGuess,
            Path imageSavePath) {
        List<MatOfPoint2f> corner_locations =
                in.stream().map(it -> it.imagePoints).map(MatOfPoint2f::new).toList();

        List<MatOfFloat> levels = in.stream().map(it -> it.levels).map(MatOfFloat::new).toList();

        int imageWidth = (int) in.get(0).size.width;
        int imageHeight = (int) in.get(0).size.height;

        MrCalResult result =
                MrCalJNI.calibrateCamera(
                        corner_locations,
                        levels,
                        params.boardWidth,
                        params.boardHeight,
                        params.squareSize,
                        imageWidth,
                        imageHeight,
                        (fxGuess + fyGuess) / 2.0);

        levels.forEach(MatOfFloat::release);
        corner_locations.forEach(MatOfPoint2f::release);

        // intrinsics are fx fy cx cy from mrcal
        JsonMatOfDouble cameraMatrixMat =
                new JsonMatOfDouble(
                        3,
                        3,
                        CvType.CV_64FC1,
                        new double[] {
                            // fx 0 cx
                            result.intrinsics[0],
                            0,
                            result.intrinsics[2],
                            // 0 fy cy
                            0,
                            result.intrinsics[1],
                            result.intrinsics[3],
                            // 0 0 1
                            0,
                            0,
                            1
                        });
        JsonMatOfDouble distortionCoefficientsMat =
                new JsonMatOfDouble(1, 8, CvType.CV_64FC1, Arrays.copyOfRange(result.intrinsics, 4, 12));

        // We get these from the JNI (retsult.optimizedPoses), but these are subtly different from the
        // ones our code used to produce. To preserve consistency, continue to redo this math
        List<Mat> rvecs = new ArrayList<>();
        List<Mat> tvecs = new ArrayList<>();
        for (var o : in) {
            var rvec = new Mat();
            var tvec = new Mat();

            // If the calibration points contain points that are negative then we need to exclude them,
            // they are considered points that we dont want to use in calibration/solvepnp. These points
            // are required prior to this to allow mrcal to work.
            Point3[] oPoints = o.objectPoints.toArray();
            Point[] iPoints = o.imagePoints.toArray();

            List<Point3> outputOPoints = new ArrayList<Point3>();
            List<Point> outputIPoints = new ArrayList<Point>();

            for (int i = 0; i < iPoints.length; i++) {
                if (iPoints[i].x >= 0 && iPoints[i].y >= 0) {
                    outputIPoints.add(iPoints[i]);
                }
            }
            for (int i = 0; i < oPoints.length; i++) {
                if (oPoints[i].x >= 0 && oPoints[i].y >= 0 && oPoints[i].z >= 0) {
                    outputOPoints.add(oPoints[i]);
                }
            }

            o.objectPoints.fromList(outputOPoints);
            o.imagePoints.fromList(outputIPoints);

            Calib3d.solvePnP(
                    o.objectPoints,
                    o.imagePoints,
                    cameraMatrixMat.getAsMatOfDouble(),
                    distortionCoefficientsMat.getAsMatOfDouble(),
                    rvec,
                    tvec);
            rvecs.add(rvec);
            tvecs.add(tvec);
        }

        List<MatOfPoint3f> objPoints = in.stream().map(it -> it.objectPoints).toList();
        List<MatOfPoint2f> imgPts = in.stream().map(it -> it.imagePoints).toList();
        List<BoardObservation> observations =
                createObservations(
                        in,
                        cameraMatrixMat.getAsMatOfDouble(),
                        distortionCoefficientsMat.getAsMatOfDouble(),
                        rvecs,
                        tvecs,
                        result.cornersUsed,
                        new double[] {result.warp_x, result.warp_y},
                        objPoints,
                        imgPts,
                        imageSavePath,
                        CameraLensModel.LENSMODEL_OPENCV);

        rvecs.forEach(Mat::release);
        tvecs.forEach(Mat::release);

        return new CameraCalibrationCoefficients(
                in.get(0).size,
                cameraMatrixMat,
                distortionCoefficientsMat,
                new double[] {result.warp_x, result.warp_y},
                observations,
                new Size(params.boardWidth, params.boardHeight),
                params.squareSize,
                CameraLensModel.LENSMODEL_OPENCV);
    }

    private List<BoardObservation> createObservations(
            List<FindBoardCornersPipe.FindBoardCornersPipeResult> in,
            Mat cameraMatrix_,
            Mat distortionCoefficients_,
            List<Mat> rvecs,
            List<Mat> tvecs,
            List<boolean[]> cornersUsed,
            double[] calobject_warp,
            List<MatOfPoint3f> objPoints,
            List<MatOfPoint2f> imgPts,
            Path imageSavePath,
            CameraLensModel lensModel) {
        // Clear the calibration image folder of any old images before we save the new ones.
        try {
            FileUtils.cleanDirectory(imageSavePath.toFile());
        } catch (Exception e) {
            logger.error("Failed to clean calibration image directory", e);
        }

        // For each observation, calc reprojection error
        Mat jac_temp = new Mat();
        List<BoardObservation> observations = new ArrayList<>();
        for (int snapshotId = 0; snapshotId < objPoints.size(); snapshotId++) {
            // Copy object points to a new mat to allow warp modification without affecting underlying
            // data
            MatOfPoint3f i_objPtsNative = new MatOfPoint3f();
            objPoints.get(snapshotId).copyTo(i_objPtsNative);

            List<Point> i_imgPts = imgPts.get(snapshotId).toList();

            if (i_objPtsNative.rows() != i_imgPts.size()) {
                throw new RuntimeException(
                        "Objpts size ("
                                + i_objPtsNative.rows()
                                + ") != imgpts size ("
                                + i_imgPts.size()
                                + ") for snapshot "
                                + snapshotId
                                + "!");
            }

            // Apply warp, if set
            if (calobject_warp != null && calobject_warp.length == 2) {
                // mrcal warp model!
                // The chessboard spans [-1, 1] on the x and y axies. We then let
                // z=k_x(1-x^2)+k_y(1-y^2)

                double xmin = 0;
                double ymin = 0;
                double xmax = params.boardWidth * params.squareSize;
                double ymax = params.boardHeight * params.squareSize;
                double k_x = calobject_warp[0];
                double k_y = calobject_warp[1];

                // Convert to list, remap z, and back to cv::Mat
                var list = i_objPtsNative.toArray();
                for (var pt : list) {
                    double x_norm = MathUtils.map(pt.x, xmin, xmax, -1, 1);
                    double y_norm = MathUtils.map(pt.y, ymin, ymax, -1, 1);
                    pt.z = k_x * (1 - x_norm * x_norm) + k_y * (1 - y_norm * y_norm);
                }
                i_objPtsNative.fromArray(list);
            }

            // Project distorted object points to image space
            var img_pts_reprojected = new MatOfPoint2f();
            Mat rvecForProjection = normalizePoseVector(rvecs.get(snapshotId));
            Mat tvecForProjection = normalizePoseVector(tvecs.get(snapshotId));
            if (rvecForProjection == null || tvecForProjection == null) {
                if (rvecForProjection != null) {
                    rvecForProjection.release();
                }
                if (tvecForProjection != null) {
                    tvecForProjection.release();
                }
                logger.debug("Skipping snapshot " + snapshotId + " due to invalid pose vector(s).");
                continue;
            }

            boolean projected = false;
            try {
                if (lensModel == CameraLensModel.LENSMODEL_OPENCV_FISHEYE) {
                    var fisheyeDistAsMatOfDouble = new MatOfDouble();
                    try {
                        distortionCoefficients_.copyTo(fisheyeDistAsMatOfDouble);
                        Calib3d.fisheye_projectPoints(
                                i_objPtsNative,
                                img_pts_reprojected,
                                rvecForProjection,
                                tvecForProjection,
                                cameraMatrix_,
                                fisheyeDistAsMatOfDouble);
                    } finally {
                        fisheyeDistAsMatOfDouble.release();
                    }
                    projected = true;
                } else {
                    Calib3d.projectPoints(
                            i_objPtsNative,
                            rvecForProjection,
                            tvecForProjection,
                            cameraMatrix_,
                            (MatOfDouble) distortionCoefficients_,
                            img_pts_reprojected,
                            jac_temp,
                            0.0);
                    projected = true;
                }
            } catch (Exception e) {
                e.printStackTrace();
                rvecForProjection.release();
                tvecForProjection.release();
                continue;
            }
            if (!projected) {
                rvecForProjection.release();
                tvecForProjection.release();
                continue;
            }

            // Calculate reprojection error for each point
            var reprojectionError = new ArrayList<Point>();
            var img_pts_reprojected_list = img_pts_reprojected.toList();
            for (int j = 0; j < img_pts_reprojected_list.size(); j++) {
                // Outliers are not part of the calibration, so don't calculate error for them
                if (!cornersUsed.get(snapshotId)[j]) {
                    continue;
                }

                // error = (measured - expected)
                var measured = img_pts_reprojected_list.get(j);
                var expected = i_imgPts.get(j);

                // Some solved poses can legitimately reproject off-image.
                // Treat these points as invalid samples, not fatal calibration errors.
                if (!Double.isFinite(measured.x)
                        || !Double.isFinite(measured.y)
                        || !Double.isFinite(expected.x)
                        || !Double.isFinite(expected.y)
                        || measured.x < 0
                        || measured.y < 0
                        || expected.x < 0
                        || expected.y < 0) {
                    continue;
                }

                var error = new Point(measured.x - expected.x, measured.y - expected.y);
                reprojectionError.add(error);
            }

            if (reprojectionError.isEmpty()) {
                rvecForProjection.release();
                tvecForProjection.release();
                continue;
            }

            var camToBoard = MathUtils.opencvRTtoPose3d(rvecForProjection, tvecForProjection);

            var inputImage = in.get(snapshotId).inputImage;
            Path image_path = null;
            String snapshotName = "img" + snapshotId + ".png";
            if (inputImage != null) {
                image_path = Paths.get(imageSavePath.toString(), snapshotName);
                Imgcodecs.imwrite(image_path.toString(), inputImage);
            }

            observations.add(
                    new BoardObservation(
                            i_objPtsNative.toList(),
                            i_imgPts,
                            reprojectionError,
                            camToBoard,
                            cornersUsed.get(snapshotId),
                            snapshotName,
                            image_path));
            rvecForProjection.release();
            tvecForProjection.release();
        }
        jac_temp.release();

        return observations;
    }

    private Mat normalizePoseVector(Mat vector) {
        if (vector == null || vector.empty()) {
            return null;
        }

        int total = (int) (vector.total() * vector.channels());
        if (total < 3) {
            return null;
        }

        Mat vector64 = new Mat();
        vector.convertTo(vector64, CvType.CV_64F);
        int total64 = (int) (vector64.total() * vector64.channels());
        if (total64 < 3) {
            vector64.release();
            return null;
        }

        double[] flat = new double[total64];
        vector64.get(0, 0, flat);
        Mat normalized = new Mat(3, 1, CvType.CV_64F);
        normalized.put(0, 0, new double[] {flat[0], flat[1], flat[2]});
        vector64.release();
        return normalized;
    }

    /** Delete all rows of mats where level is < 0. Useful for opencv */
    private void deleteIgnoredPoints(
            MatOfPoint3f objPtsMatIn,
            MatOfPoint2f imgPtsMatIn,
            MatOfFloat levelsMat,
            MatOfPoint3f objPtsMatOut,
            MatOfPoint2f imgPtsMatOut) {
        var levels = levelsMat.toArray();
        var objPtsIn = objPtsMatIn.toArray();
        var imgPtsIn = imgPtsMatIn.toArray();

        var objPtsOut = new ArrayList<Point3>();
        var imgPtsOut = new ArrayList<Point>();

        for (int i = 0; i < levels.length; i++) {
            if (levels[i] >= 0) {
                // point survives
                objPtsOut.add(objPtsIn[i]);
                imgPtsOut.add(imgPtsIn[i]);
            }
        }

        objPtsMatOut.fromList(objPtsOut);
        imgPtsMatOut.fromList(imgPtsOut);
    }

    public static class CalibratePipeParams {
        // Size (in # of corners) of the calibration object
        public int boardHeight;
        public int boardWidth;
        // And size of each square
        public double squareSize;

        public boolean useMrCal;
        public CameraLensModel lensModel;

        public CalibratePipeParams(
                int boardHeightSquares,
                int boardWidthSquares,
                double squareSize,
                boolean usemrcal,
                CameraLensModel lensModel) {
            this.boardHeight = boardHeightSquares - 1;
            this.boardWidth = boardWidthSquares - 1;
            this.squareSize = squareSize;
            this.useMrCal = usemrcal;
            this.lensModel = lensModel != null ? lensModel : CameraLensModel.LENSMODEL_OPENCV;
        }
    }
}
