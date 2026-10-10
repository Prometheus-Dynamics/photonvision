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
package org.photonvision.vision.frame;

import org.photonvision.styx.StyxResult;
import org.photonvision.vision.opencv.CVMat;

/**
 * A frame from the Styx backend. The pixels stay in Styx: the frame carries the detector's result
 * for this frame and the camera handle, and its Mats are empty (no OpenCV on this path).
 */
public class StyxFrame extends Frame {
    public final long handle;
    public final StyxResult result;

    public StyxFrame(
            long sequenceID,
            long handle,
            StyxResult result,
            long timestampNanos,
            FrameStaticProperties frameStaticProperties) {
        super(
                sequenceID,
                new CVMat(),
                new CVMat(),
                FrameThresholdType.NONE,
                timestampNanos,
                frameStaticProperties);
        this.handle = handle;
        this.result = result;
    }
}
