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

import edu.wpi.first.cscore.VideoMode;
import edu.wpi.first.util.PixelFormat;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import org.opencv.core.Size;
import org.photonvision.common.configuration.CameraConfiguration;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.styx.StyxCameras;
import org.photonvision.styx.StyxJNI;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.JsonMatOfDouble;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.camera.csi.LibcameraGpuSource.FPSRatedVideoMode;
import org.photonvision.vision.opencv.ImageRotationMode;
import org.photonvision.vision.processes.VisionSourceSettables;

/**
 * Settings of a CSI camera on the Styx backend. A video mode is a Styx capture at an exact size and
 * rate, so changing it reopens the camera session. Controls map to Styx's standard controls.
 */
public class StyxSettables extends VisionSourceSettables {
    private static final Logger logger = new Logger(StyxSettables.class, LogGroup.Camera);

    /**
     * The Raze OV9782 modes (grey, the sensor's 3A-processed luma), opened at exactly this rate: the
     * sensor's three native modes (Styx's ov9782.toml: the full array, its top 720 lines, and the
     * array subsampled 2x2), up to 120.6, 144.2 and 259.8 fps (exact rates). The Raze image drives the sensor
     * through Styx's sensor bridge, which sets the sensor's timing itself; exact rates and 640x400
     * were only unreliable on the kernel-driven sensor (ov9282 driver), which the image no longer
     * binds.
     */
    private static final int[][] MODES = {
        {1280, 800, 30},
        {1280, 800, 60},
        {1280, 800, 120},
        {1280, 720, 30},
        {1280, 720, 60},
        {1280, 720, 120},
        {1280, 720, 144},
        {640, 400, 30},
        {640, 400, 60},
        {640, 400, 120},
        {640, 400, 240},
        // The centred 1280x480 window (Styx's ov9282 module): full pixel pitch, 480 of the 800 rows,
        // so readout is about 40% shorter. Calibrations of the full field apply, shifted by the window.
        {1280, 480, 30},
        {1280, 480, 60},
        {1280, 480, 120},
        {1280, 480, 200},
    };

    /** The opened mode's window in the full field ({x, y, width, height}); empty for the full field. */
    private int[] crop = new int[0];

    private FPSRatedVideoMode currentVideoMode;
    /**
     * The picture's rotation. Styx turns the picture on the sensor: 180 degrees is both flips, 0 is
     * none. 90 and 270 degrees are not supported on this backend and leave the picture unturned, as
     * the libcamera path does.
     */
    private ImageRotationMode rotation = ImageRotationMode.DEG_0;

    private long handle = 0;
    private double lastExposureUs = 20000;
    private boolean lastAutoExposure = false;
    private int lastGain = 50;
    private boolean lastAwb = true;
    private double lastColourTemp = 5000;
    private double lastRedGain = 1.1;
    private double lastBlueGain = 2.0;

    public final Object CAMERA_LOCK = new Object();

    public StyxSettables(CameraConfiguration configuration) {
        super(configuration);
        try {
            StyxJNI.load();
        } catch (IOException | UnsatisfiedLinkError e) {
            logger.error("Unable to load libphotonstyx", e);
        }
        videoModes = new HashMap<>();
        for (int i = 0; i < MODES.length; i++) {
            int[] m = MODES[i];
            videoModes.put(i, new FPSRatedVideoMode(PixelFormat.kGray, m[0], m[1], m[2], m[2], 1));
        }
        currentVideoMode = (FPSRatedVideoMode) videoModes.get(0);
        this.cameraPropertiesCached = true;
    }

    /** The Styx session handle, or 0 while the camera is closed. */
    public long handle() {
        synchronized (CAMERA_LOCK) {
            return handle;
        }
    }

    /**
     * Turn the picture. A flip is set when the session opens, so a change that alters the flips
     * reopens the camera (under CAMERA_LOCK, as a video mode change does). The picture's size and
     * its calibration are unchanged: a calibration is taken in the orientation it is used in.
     */
    public void setRotation(ImageRotationMode mode) {
        if (mode == rotation) return;
        boolean wasFlipped = isFlipped();
        rotation = mode;
        if (mode == ImageRotationMode.DEG_90_CCW || mode == ImageRotationMode.DEG_270_CCW) {
            logger.warn(
                    "Styx cameras support 0 and 180 degree rotation; "
                            + mode
                            + " is not applied on "
                            + getConfiguration().nickname);
        }
        if (isFlipped() != wasFlipped) {
            setVideoModeInternal(getCurrentVideoMode());
        }
    }

    public ImageRotationMode getRotation() {
        return rotation;
    }

    /** Whether the sensor is turned 180 degrees (both flips). */
    private boolean isFlipped() {
        return rotation == ImageRotationMode.DEG_180_CCW;
    }

    /** Close the session (the camera is released). */
    public void close() {
        synchronized (CAMERA_LOCK) {
            if (handle != 0) {
                StyxJNI.close(handle);
                handle = 0;
            }
        }
    }

    @Override
    public FPSRatedVideoMode getCurrentVideoMode() {
        return currentVideoMode;
    }

    @Override
    protected void setVideoModeInternal(VideoMode videoMode) {
        var mode = (FPSRatedVideoMode) videoMode;
        synchronized (CAMERA_LOCK) {
            close();
            String styxName = StyxCameras.resolve(getConfiguration().matchedCameraInfo.path());
            if (styxName == null) {
                logger.error(
                        "No Styx camera matches " + getConfiguration().matchedCameraInfo.path());
                handle = 0;
                currentVideoMode = mode;
                return;
            }
            try {
                handle =
                        StyxJNI.open(
                                styxName,
                                mode.width,
                                mode.height,
                                mode.fps,
                                isFlipped(),
                                isFlipped());
                crop = StyxJNI.cropRect(handle);
                logger.info(
                        "Opened Styx camera "
                                + getConfiguration().nickname
                                + " at "
                                + mode.width
                                + "x"
                                + mode.height
                                + "@"
                                + mode.fps);
            } catch (RuntimeException e) {
                logger.error("Could not open the Styx camera in mode " + mode.width + "x" + mode.height, e);
                handle = 0;
            }
        }
        currentVideoMode = mode;
        // The session starts with its defaults: reapply what the user set.
        setAutoExposure(lastAutoExposure);
        if (!lastAutoExposure) setExposureRaw(lastExposureUs);
        setGain(lastGain);
        setAutoWhiteBalance(lastAwb);
        if (!lastAwb) setWhiteBalanceTemp(lastColourTemp);
        setRedGain((int) Math.round(lastRedGain * 10));
        setBlueGain((int) Math.round(lastBlueGain * 10));
    }

    @Override
    public HashMap<Integer, VideoMode> getAllVideoModes() {
        return videoModes;
    }

    /**
     * A windowed mode (e.g. 1280x480) without a calibration of its own uses the full field's: the
     * same lens, so the same focal lengths and distortion, with the principal point moved by the
     * window's offset (cx - x, cy - y).
     */
    @Override
    protected void calculateFrameStaticProps() {
        super.calculateFrameStaticProps();
        var mode = getCurrentVideoMode();
        if (frameStaticProperties.cameraCalibration != null || crop.length != 4 || mode == null) {
            return;
        }
        int x = crop[0], y = crop[1];
        var full =
                getConfiguration().calibrations.stream()
                        .filter(
                                c ->
                                        c.unrotatedImageSize.width >= x + mode.width
                                                && c.unrotatedImageSize.height >= y + mode.height
                                                && (c.unrotatedImageSize.width > mode.width
                                                        || c.unrotatedImageSize.height > mode.height))
                        .findFirst()
                        .orElse(null);
        if (full == null) return;
        double[] k = full.cameraIntrinsics.data.clone(); // row-major 3x3
        k[2] -= x;
        k[5] -= y;
        var windowed =
                new CameraCalibrationCoefficients(
                        new Size(mode.width, mode.height),
                        new JsonMatOfDouble(
                                full.cameraIntrinsics.rows,
                                full.cameraIntrinsics.cols,
                                full.cameraIntrinsics.type,
                                k),
                        full.distCoeffs,
                        full.calobjectWarp,
                        List.of(),
                        full.calobjectSize,
                        full.calobjectSpacing,
                        full.lensmodel);
        frameStaticProperties = new FrameStaticProperties(mode, getFOV(), windowed);
    }

    @Override
    public void setExposureRaw(double exposureRaw) {
        lastExposureUs = exposureRaw;
        setControl(StyxJNI.CONTROL_AE, 0);
        double us = Math.max(getMinExposureRaw(), Math.min(getMaxExposureRaw(), exposureRaw));
        setControl(StyxJNI.CONTROL_EXPOSURE_US, us);
    }

    @Override
    public double getMinExposureRaw() {
        return 7;
    }

    @Override
    public double getMaxExposureRaw() {
        return 80000;
    }

    @Override
    public void setAutoExposure(boolean cameraAutoExposure) {
        lastAutoExposure = cameraAutoExposure;
        setControl(StyxJNI.CONTROL_AE, cameraAutoExposure ? 1 : 0);
    }

    @Override
    public void setWhiteBalanceTemp(double temp) {
        lastColourTemp = temp;
        setControl(StyxJNI.CONTROL_AWB, 0);
        setControl(StyxJNI.CONTROL_COLOUR_TEMPERATURE, temp);
    }

    @Override
    public void setAutoWhiteBalance(boolean autowb) {
        lastAwb = autowb;
        setControl(StyxJNI.CONTROL_AWB, autowb ? 1 : 0);
    }

    @Override
    public double getMinWhiteBalanceTemp() {
        return 2500;
    }

    @Override
    public double getMaxWhiteBalanceTemp() {
        return 8000;
    }

    @Override
    public void setBrightness(int brightness) {
        // Styx has no brightness control; the 3A loop owns the image level. Exposure compensation
        // (EV) is the closest and is left to the user's exposure settings.
        logger.debug("Brightness is not a Styx control; ignored");
    }

    @Override
    public void setGain(int gain) {
        lastGain = gain;
        // The UI's 0-100 maps to a total gain of 1 to 10, as on the libcamera path.
        setControl(
                StyxJNI.CONTROL_GAIN,
                Math.max(1.0, Math.min(10.0, MathUtils.map(gain, 0.0, 100.0, 1.0, 10.0))));
    }

    @Override
    public void setRedGain(int red) {
        lastRedGain = red / 10.0;
        setControl(StyxJNI.CONTROL_RED_GAIN, lastRedGain);
    }

    @Override
    public void setBlueGain(int blue) {
        lastBlueGain = blue / 10.0;
        setControl(StyxJNI.CONTROL_BLUE_GAIN, lastBlueGain);
    }

    private void setControl(int kind, double value) {
        synchronized (CAMERA_LOCK) {
            if (handle == 0) return;
            try {
                StyxJNI.setControl(handle, kind, value);
            } catch (RuntimeException e) {
                logger.warn("Styx control " + kind + " = " + value + " failed: " + e.getMessage());
            }
        }
    }
}
