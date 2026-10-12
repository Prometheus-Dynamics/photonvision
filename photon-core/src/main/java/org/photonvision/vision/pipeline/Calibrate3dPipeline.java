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

package org.photonvision.vision.pipeline;

import edu.wpi.first.math.Pair;
import edu.wpi.first.math.util.Units;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.photonvision.common.dataflow.DataChangeService;
import org.photonvision.common.dataflow.events.OutgoingUIEvent;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.SerializationUtils;
import org.photonvision.styx.StyxBackend;
import org.photonvision.styx.StyxCalibration;
import org.photonvision.styx.StyxJNI;
import org.photonvision.styx.StyxResult;
import org.photonvision.vision.calibration.BoardObservation;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.frame.StyxFrame;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.ImageRotationMode;
import org.photonvision.vision.pipe.CVPipe.CVPipeResult;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipe.impl.Calibrate3dPipe;
import org.photonvision.vision.pipe.impl.Calibrate3dPipe.CalibrationInput;
import org.photonvision.vision.pipe.impl.FindBoardCornersPipe;
import org.photonvision.vision.pipe.impl.FindBoardCornersPipe.FindBoardCornersPipeResult;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.pipeline.result.CalibrationPipelineResult;

public class Calibrate3dPipeline
        extends CVPipeline<CVPipelineResult, Calibration3dPipelineSettings> {
    // For logging
    private static final Logger logger = new Logger(Calibrate3dPipeline.class, LogGroup.General);

    // Find board corners decides internally between opencv and mrgingham
    private final FindBoardCornersPipe findBoardCornersPipe = new FindBoardCornersPipe();
    private final Calibrate3dPipe calibrate3dPipe = new Calibrate3dPipe();
    private final CalculateFPSPipe calculateFPSPipe = new CalculateFPSPipe();

    // Getter methods have been set for calibrate and takeSnapshot
    private boolean takeSnapshot = false;

    // Output of the corners
    public final List<FindBoardCornersPipeResult> foundCornersList;

    /// Output of the calibration, getter method is set for this.
    private CVPipeResult<CameraCalibrationCoefficients> calibrationOutput;

    private final int minSnapshots;
    // Styx backend (no Mats): snapshots are corner ids and pixels, found in the camera session and
    // solved by Eidos. A snapshot is pending from its request until the next camera frame carries it.
    private final List<StyxCalibration.View> styxViews = new ArrayList<>();
    private boolean styxSnapshotPending = false;
    private CameraCalibrationCoefficients styxCalibration;

    private boolean calibrating = false;

    private static final FrameThresholdType PROCESSING_TYPE = FrameThresholdType.NONE;

    public Calibrate3dPipeline() {
        this(12);
    }

    public Calibrate3dPipeline(int minSnapshots) {
        super(PROCESSING_TYPE);
        this.settings = new Calibration3dPipelineSettings();
        this.foundCornersList = new ArrayList<>();
        this.minSnapshots = minSnapshots;
    }

    @Override
    protected void setPipeParamsImpl() {
        if (StyxBackend.enabled()) {
            // The OpenCV board finder and solver are not used on the Styx backend (see processStyx).
            return;
        }
        findBoardCornersPipe.setParams(
                new FindBoardCornersPipe.FindCornersPipeParams(
                        settings.boardHeight,
                        settings.boardWidth,
                        settings.boardType,
                        settings.tagFamily,
                        settings.gridSize,
                        settings.markerSize,
                        settings.streamingFrameDivisor,
                        settings.useOldPattern));

        calibrate3dPipe.setParams(
                new Calibrate3dPipe.CalibratePipeParams(
                        settings.boardHeight,
                        settings.boardWidth,
                        settings.gridSize,
                        settings.useMrCal,
                        settings.lensModel));
    }

    @Override
    protected CVPipelineResult process(Frame frame, Calibration3dPipelineSettings settings) {
        if (frame instanceof StyxFrame styxFrame) {
            return processStyx(styxFrame);
        }
        Mat inputColorMat = frame.colorImage.getMat();

        if (this.calibrating || inputColorMat.empty()) {
            return new CVPipelineResult(frame.sequenceID, 0, 0, null, frame);
        }

        if (getSettings().inputImageRotationMode != ImageRotationMode.DEG_0) {
            // All this calibration assumes zero rotation. If we want a rotation, it should
            // be applied at
            // the output
            logger.error(
                    "Input image rotation was non-zero! Calibration wasn't designed to deal with this. Attempting to manually change back to zero");
            getSettings().inputImageRotationMode = ImageRotationMode.DEG_0;
            return new CVPipelineResult(frame.sequenceID, 0, 0, List.of(), frame);
        }

        long sumPipeNanosElapsed = 0L;

        // Check if the frame has chessboard corners
        var outputColorCVMat = new CVMat();
        inputColorMat.copyTo(outputColorCVMat.getMat());

        FindBoardCornersPipeResult findBoardResult =
                findBoardCornersPipe.run(Pair.of(inputColorMat, outputColorCVMat.getMat())).output;

        if (takeSnapshot) {
            // Set snapshot to false even if we don't find a board
            takeSnapshot = false;

            if (findBoardResult != null) {
                // Only copy the image into the result when we absolutely must
                findBoardResult.inputImage = inputColorMat.clone();

                foundCornersList.add(findBoardResult);

                // update the UI
                broadcastState();
            }
        }

        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;

        frame.release();

        // Return the drawn chessboard if corners are found, if not, then return the
        // input image.
        return new CalibrationPipelineResult(
                frame.sequenceID,
                sumPipeNanosElapsed,
                fps, // Unused but here in case
                new Frame(
                        frame.sequenceID,
                        new CVMat(),
                        outputColorCVMat,
                        FrameThresholdType.NONE,
                        frame.frameStaticProperties),
                getCornersList());
    }

    /**
     * Styx backend: the board is found in the camera session. A snapshot asks for the board on the
     * next frame, and that frame's record says whether the board was seen. No pixels reach Java, so
     * the output is the frame itself (its streams are served from the camera session).
     */
    private CVPipelineResult processStyx(StyxFrame frame) {
        if (calibrating) {
            return new CVPipelineResult(frame.sequenceID, 0, 0, List.of(), frame);
        }
        if (takeSnapshot) {
            // Set snapshot to false even if we don't find a board
            takeSnapshot = false;
            try {
                StyxJNI.requestBoard(frame.handle, styxBoardJson());
                styxSnapshotPending = true;
            } catch (IllegalArgumentException e) {
                logger.error("Cannot look for the calibration board: " + e.getMessage());
            }
        } else if (styxSnapshotPending && frame.result.board() != null) {
            styxSnapshotPending = false;
            StyxResult.Board board = frame.result.board();
            if (board.found()) {
                styxViews.add(StyxCalibration.View.of(board));
                broadcastState();
            } else {
                logger.info("No calibration board found in the snapshot");
            }
        }
        return new CVPipelineResult(frame.sequenceID, 0, 0, List.of(), frame);
    }

    private String styxBoardJson() {
        var s = getSettings();
        return StyxCalibration.boardJson(
                s.boardType,
                s.boardWidth,
                s.boardHeight,
                s.gridSize,
                s.markerSize,
                s.tagFamily,
                s.useOldPattern);
    }

    /** Styx backend: solve the camera over the snapshots with Eidos. Null when that fails. */
    private CameraCalibrationCoefficients tryCalibrationStyx() {
        var s = getSettings();
        styxCalibration = null;
        if (styxViews.size() < 3) {
            logger.error("Calibration needs at least 3 snapshots, have " + styxViews.size());
            return null;
        }
        this.calibrating = true;
        try {
            var solved = StyxCalibration.solve(styxBoardJson(), s.lensModel, styxViews);
            if (!solved.converged()) {
                logger.warn("Eidos calibration did not converge; the result may be poor");
            }
            styxCalibration =
                    StyxCalibration.toCoefficients(
                            solved,
                            styxViews,
                            s.resolution,
                            s.lensModel,
                            s.boardWidth,
                            s.boardHeight,
                            s.gridSize);
            logger.info(
                    "CALIBRATION SUCCESS (Eidos) with "
                            + styxViews.size()
                            + " snapshots, rms "
                            + solved.rms()
                            + " px, fx "
                            + solved.fx()
                            + ", fy "
                            + solved.fy());
            return styxCalibration;
        } catch (RuntimeException e) {
            logger.error("Calibration failed: " + e.getMessage());
            return null;
        } finally {
            this.calibrating = false;
        }
    }

    List<List<Point>> getCornersList() {
        return foundCornersList.stream().map(it -> it.imagePoints.toList()).toList();
    }

    public boolean hasEnough() {
        return snapshotCount() >= minSnapshots;
    }

    public CameraCalibrationCoefficients tryCalibration(Path imageSavePath) {
        if (StyxBackend.enabled()) {
            return tryCalibrationStyx();
        }

        if (!hasEnough()) {
            logger.info(
                    "Not enough snapshots! Only got "
                            + snapshotCount()
                            + " of "
                            + minSnapshots
                            + " -- returning null..");
            return null;
        }

        this.calibrating = true;

        /*
         * Pass the board corners to the pipe, which will check again to see if all
         * boards are valid
         * and returns the corresponding image and object points
         */
        calibrationOutput =
                calibrate3dPipe.run(
                        new CalibrationInput(foundCornersList, frameStaticProperties, imageSavePath));

        this.calibrating = false;

        return calibrationOutput.output;
    }

    public void takeSnapshot() {
        takeSnapshot = true;
    }

    public List<BoardObservation> perViewErrors() {
        if (StyxBackend.enabled()) {
            return styxCalibration == null ? List.of() : styxCalibration.observations;
        }
        return calibrationOutput.output.observations;
    }

    public void finishCalibration() {
        foundCornersList.forEach(it -> it.release());
        foundCornersList.clear();
        styxViews.clear();
        styxSnapshotPending = false;

        broadcastState();
    }

    public void broadcastState() {
        var state =
                SerializationUtils.objectToHashMap(
                        new UICalibrationData(
                                snapshotCount(),
                                settings.cameraVideoModeIndex,
                                minSnapshots,
                                hasEnough(),
                                Units.metersToInches(settings.gridSize),
                                Units.metersToInches(settings.markerSize),
                                settings.boardWidth,
                                settings.boardHeight,
                                settings.boardType,
                                settings.useOldPattern,
                                settings.tagFamily,
                                settings.lensModel));

        DataChangeService.getInstance()
                .publishEvent(OutgoingUIEvent.wrappedOf("calibrationData", state));
    }

    private int snapshotCount() {
        return StyxBackend.enabled() ? styxViews.size() : foundCornersList.size();
    }

    public boolean removeSnapshot(int index) {
        if (StyxBackend.enabled()) {
            if (index < 0 || index >= styxViews.size()) {
                logger.error("Could not remove snapshot at index " + index);
                return false;
            }
            styxViews.remove(index);
            return true;
        }
        try {
            foundCornersList.remove(index);
            return true;
        } catch (ArrayIndexOutOfBoundsException e) {
            logger.error("Could not remove snapshot at index " + index, e);
            return false;
        }
    }

    public CameraCalibrationCoefficients cameraCalibrationCoefficients() {
        if (StyxBackend.enabled()) {
            return styxCalibration;
        }
        return calibrationOutput.output;
    }

    @Override
    public void release() {
        foundCornersList.forEach(it -> it.release());
        foundCornersList.clear();
        styxViews.clear();
        styxSnapshotPending = false;

        findBoardCornersPipe.release();
        calibrate3dPipe.release();
        calculateFPSPipe.release();
        super.release();
    }
}
