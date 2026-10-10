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

import org.photonvision.vision.apriltag.AprilTagFamily;

/**
 * Settings of the Eidos AprilTag pipeline (Styx backend only): the family, the multi-tag solve and
 * the filters the Eidos result can apply. Decision margins are not produced by Eidos, so only the
 * hamming filter applies.
 */
public class EidosAprilTagPipelineSettings extends AdvancedPipelineSettings {
    public AprilTagFamily tagFamily = AprilTagFamily.kTag36h11;
    public int hammingDist = 0;
    public boolean doMultiTarget = false;
    public boolean doSingleTargetAlways = false;
    /** Full-frame search at least every this many frames (Eidos tracking). */
    public int fullSearchEvery = 8;
    /** Encode the processed stream (grey frame with the detected tags outlined). */
    public boolean processedStream = false;

    public EidosAprilTagPipelineSettings() {
        super();
        pipelineType = PipelineType.EidosAprilTag;
        cameraAutoExposure = false;
        ledMode = false;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = super.hashCode();
        result = prime * result + ((tagFamily == null) ? 0 : tagFamily.hashCode());
        result = prime * result + hammingDist;
        result = prime * result + (doMultiTarget ? 1231 : 1237);
        result = prime * result + (doSingleTargetAlways ? 1231 : 1237);
        result = prime * result + fullSearchEvery;
        result = prime * result + (processedStream ? 1231 : 1237);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!super.equals(obj)) return false;
        if (getClass() != obj.getClass()) return false;
        EidosAprilTagPipelineSettings other = (EidosAprilTagPipelineSettings) obj;
        return tagFamily == other.tagFamily
                && hammingDist == other.hammingDist
                && doMultiTarget == other.doMultiTarget
                && doSingleTargetAlways == other.doSingleTargetAlways
                && fullSearchEvery == other.fullSearchEvery
                && processedStream == other.processedStream;
    }
}
