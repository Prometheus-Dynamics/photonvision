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
package org.photonvision.vision.camera.csi;

import org.photonvision.common.configuration.CameraConfiguration;
import org.photonvision.vision.camera.CameraType;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.FrameProvider;
import org.photonvision.vision.frame.provider.StyxFrameProvider;
import org.photonvision.vision.processes.VisionSource;
import org.photonvision.vision.processes.VisionSourceSettables;

/** A CSI camera on the Styx backend (no libcamera): see {@link StyxSettables}. */
public class StyxCameraSource extends VisionSource {
    private final StyxSettables settables;
    private StyxFrameProvider frameProvider;

    public StyxCameraSource(CameraConfiguration configuration) {
        super(configuration);
        if (configuration.matchedCameraInfo.type() != CameraType.ZeroCopyPicam) {
            throw new IllegalArgumentException("StyxCameraSource only accepts CSI (ZeroCopyPicam) cameras");
        }
        // The same quirks as a libcamera CSI camera (LibcameraGpuSource): a fresh configuration
        // (a camera assigned from the UI) has none, and VisionModule reads them.
        if (configuration.cameraQuirks == null) configuration.cameraQuirks = QuirkyCamera.ZeroCopyPiCamera;
        settables = new StyxSettables(configuration);
        frameProvider = new StyxFrameProvider(settables);
    }

    @Override
    public FrameProvider getFrameProvider() {
        return frameProvider;
    }

    @Override
    public VisionSourceSettables getSettables() {
        return settables;
    }

    @Override
    public void remakeSettables() {
        // Settables for a Styx camera are never remade.
    }

    @Override
    public boolean isVendorCamera() {
        return false;
    }

    @Override
    public boolean hasLEDs() {
        return false;
    }

    @Override
    public void release() {
        frameProvider.release();
        frameProvider = null;
    }
}
