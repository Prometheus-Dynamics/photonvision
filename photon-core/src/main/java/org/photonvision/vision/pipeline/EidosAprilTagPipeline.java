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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.styx.StyxJNI;
import org.photonvision.styx.StyxResult;
import org.photonvision.styx.StyxTargetMath;
import org.photonvision.targeting.MultiTargetPNPResult;
import org.photonvision.targeting.PhotonTrackedTarget;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.frame.StyxFrame;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipeline.result.CVPipelineResult;

/**
 * AprilTag detection on the Styx backend: Eidos detects the tags inside the camera's Styx session
 * (see photon-styx), and this pipeline turns the result into PhotonVision's targets. It never
 * touches the frame's pixels, so it works without OpenCV.
 */
public class EidosAprilTagPipeline
        extends CVPipeline<CVPipelineResult, EidosAprilTagPipelineSettings> {
    private static final Logger logger = new Logger(EidosAprilTagPipeline.class, LogGroup.VisionModule);
    private static final FrameThresholdType PROCESSING_TYPE = FrameThresholdType.NONE;
    private static final int MAX_TARGETS = 64;

    /** The last configuration sent to the native session, so it is sent only on change. */
    private String sentConfig = null;
    private long sentHandle = 0;

    private final CalculateFPSPipe calculateFPSPipe = new CalculateFPSPipe();

    public EidosAprilTagPipeline() {
        super(PROCESSING_TYPE);
        settings = new EidosAprilTagPipelineSettings();
    }

    public EidosAprilTagPipeline(EidosAprilTagPipelineSettings settings) {
        super(PROCESSING_TYPE);
        this.settings = settings;
    }

    @Override
    protected void setPipeParamsImpl() {
        // Nothing to precompute: the configuration is built in process() from the frame's camera.
    }

    @Override
    protected CVPipelineResult process(Frame frame, EidosAprilTagPipelineSettings settings) {
        long started = System.nanoTime();
        if (!(frame instanceof StyxFrame styxFrame) || styxFrame.result == null) {
            // No frame from the camera this tick.
            return new CVPipelineResult(frame.sequenceID, 0, 0, List.of(), frame);
        }
        // Sends the configuration to the camera session when it changed. It takes effect on the
        // next frame the session detects (one frame of latency after a settings change).
        String json = EidosDetectorConfig.json(settings, frame.frameStaticProperties);
        if (!json.equals(sentConfig) || sentHandle != styxFrame.handle) {
            try {
                StyxJNI.configure(styxFrame.handle, json);
                sentConfig = json;
                sentHandle = styxFrame.handle;
            } catch (RuntimeException e) {
                logger.error("Could not configure the Eidos detector", e);
            }
        }

        StyxResult result = styxFrame.result;
        var intrinsics = intrinsics(frame);
        // Poses only in 3D mode, as PhotonVision's AprilTag pipeline (and only with a calibration,
        // without which Eidos solves none).
        boolean withPose = settings.solvePNPEnabled;
        List<PhotonTrackedTarget> targets = new ArrayList<>();
        for (StyxResult.Target t : result.targets()) {
            if (t.hamming() > settings.hammingDist) continue;
            targets.add(StyxTargetMath.toPhotonTarget(t, intrinsics, withPose));
        }
        if (targets.size() > MAX_TARGETS) {
            targets = targets.subList(0, MAX_TARGETS);
        }

        Optional<MultiTargetPNPResult> multi = Optional.empty();
        if (withPose && settings.doMultiTarget && result.multiValid()) {
            multi = Optional.of(StyxTargetMath.toMultiTarget(result));
        }

        // Processing time: Eidos's detection (and pose) time in the camera session, plus this
        // conversion.
        double processingNanos = result.processingNanos() + (System.nanoTime() - started);
        int fps = calculateFPSPipe.run(null).output;
        var out = new CVPipelineResult(frame.sequenceID, processingNanos, fps, List.of(), multi, frame);
        out.photonTargets = targets;
        out.styxResult = result;
        return out;
    }

    private static StyxTargetMath.Intrinsics intrinsics(Frame frame) {
        var props = frame.frameStaticProperties;
        double width = props.imageWidth;
        double height = props.imageHeight;
        if (props.cameraCalibration != null) {
            double[] k = props.cameraCalibration.getIntrinsicsArr(); // row-major 3x3
            return new StyxTargetMath.Intrinsics(k[0], k[4], k[2], k[5], props.imageArea);
        }
        // Without calibration: PhotonVision's default 70-degree-wide guess is not used; a 60-degree
        // horizontal field of view with the centre of the image.
        double fx = width / (2 * Math.tan(Math.toRadians(30)));
        return new StyxTargetMath.Intrinsics(
                fx, fx, props.centerX, props.centerY, width * height);
    }
}
