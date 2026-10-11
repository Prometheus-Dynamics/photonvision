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

import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.pipeline.result.CVPipelineResult;

/**
 * An AprilTag pipeline on the Styx backend. The web UI only knows PhotonVision's AprilTag
 * pipeline, whose OpenCV detector has no pixels to read on a Styx camera, so on that backend an
 * AprilTag pipeline runs Eidos ({@link EidosAprilTagPipeline}) with the AprilTag settings the UI
 * edits: the tag family, hamming distance and multi-target settings carry over, and "draw on the
 * output" turns the processed stream on. The decimate, blur, threads and refinement settings are
 * OpenCV detector settings and do not apply.
 */
public class StyxAprilTagPipeline extends CVPipeline<CVPipelineResult, AprilTagPipelineSettings> {
    private final EidosAprilTagPipeline eidos = new EidosAprilTagPipeline();
    private final EidosAprilTagPipelineSettings eidosSettings = new EidosAprilTagPipelineSettings();

    public StyxAprilTagPipeline(AprilTagPipelineSettings settings) {
        super(FrameThresholdType.NONE);
        this.settings = settings;
    }

    @Override
    protected void setPipeParamsImpl() {}

    @Override
    protected CVPipelineResult process(Frame frame, AprilTagPipelineSettings settings) {
        // Read every frame, so a change in the UI applies on the next one.
        eidosSettings.tagFamily = settings.tagFamily;
        eidosSettings.hammingDist = settings.hammingDist;
        eidosSettings.doMultiTarget = settings.doMultiTarget;
        eidosSettings.doSingleTargetAlways = settings.doSingleTargetAlways;
        eidosSettings.processedStream = settings.outputShouldDraw;
        eidos.setPipeParams(frame.frameStaticProperties, eidosSettings, cameraQuirks);
        return eidos.process(frame, eidosSettings);
    }

    @Override
    public void release() {
        eidos.release();
        super.release();
    }
}
