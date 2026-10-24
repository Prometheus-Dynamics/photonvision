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
package org.photonvision.styx;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.jni.CombinedRuntimeLoader;

/**
 * JNI into libphotonstyx.so (photon-styx/ in the PhotonVision fork): Styx capture and Eidos
 * AprilTag detection for the experimental Styx backend. Handles are opaque positive longs. Only the
 * Styx backend loads this library (see {@link StyxBackend}).
 */
public final class StyxJNI {
    private static final Logger logger = new Logger(StyxJNI.class, LogGroup.Camera);
    private static boolean loaded = false;

    /** Control kinds for {@link #setControl}; mirror photon-styx/src/session.rs. */
    public static final int CONTROL_EXPOSURE_US = 1;

    public static final int CONTROL_GAIN = 2;
    public static final int CONTROL_AE = 3;
    public static final int CONTROL_EV = 4;
    public static final int CONTROL_FPS = 5;
    public static final int CONTROL_AWB = 6;
    public static final int CONTROL_COLOUR_TEMPERATURE = 7;
    public static final int CONTROL_RED_GAIN = 8;
    public static final int CONTROL_BLUE_GAIN = 9;

    private StyxJNI() {}

    /** Where the build puts the library in the jar (photon-core/build.gradle, -PstyxBackend). */
    private static final String RESOURCE = "/linux/arm64/libphotonstyx.so";

    /**
     * Load libphotonstyx: from {@code -Dphotonvision.styx.library=/path/libphotonstyx.so} when set,
     * otherwise from the jar's native resources. The library is not in the jar's
     * ResourceInformation.json (the WPILib plugin writes that from its own natives), so when
     * CombinedRuntimeLoader cannot find it the jar entry is copied to java.io.tmpdir and loaded from
     * there.
     */
    public static synchronized void load() throws IOException {
        if (loaded) return;
        String override = System.getProperty("photonvision.styx.library");
        if (override != null && !override.isBlank()) {
            System.load(override);
        } else {
            try {
                CombinedRuntimeLoader.loadLibraries(StyxJNI.class, "photonstyx");
            } catch (IOException e) {
                System.load(extractResource().toString());
            }
        }
        loaded = true;
        logger.info("Loaded libphotonstyx");
    }

    private static Path extractResource() throws IOException {
        try (InputStream in = StyxJNI.class.getResourceAsStream(RESOURCE)) {
            if (in == null) throw new IOException("Could not find library photonstyx (" + RESOURCE + ")");
            Path dir = Files.createTempDirectory("photonstyx");
            Path lib = dir.resolve("libphotonstyx.so");
            Files.copy(in, lib, StandardCopyOption.REPLACE_EXISTING);
            lib.toFile().deleteOnExit();
            dir.toFile().deleteOnExit();
            return lib;
        }
    }

    public static synchronized boolean isLoaded() {
        return loaded;
    }

    /** Camera names Styx finds (the names {@link #open} takes). */
    public static native String[] listCameras();

    /**
     * Open a camera at exactly width x height and fps, with the sensor flipped left-right
     * (hflip) and/or upside down (vflip); both is a 180 degree turn. The flips are set at open:
     * changing them means reopening. Returns a handle.
     */
    public static native long open(
            String camera, int width, int height, int fps, boolean hflip, boolean vflip);

    public static native void close(long handle);

    /**
     * Where the opened mode's frames sit in the sensor's full field, as {x, y, width, height}, or an
     * empty array when the mode is the full field.
     */
    public static native int[] cropRect(long handle);

    /** Pipeline configuration as JSON (see photon-styx/src/config.rs). */
    public static native void configure(long handle, String json);

    /** Apply a standard control (CONTROL_*); returns the value in effect (NaN if unknown). */
    public static native double setControl(long handle, int kind, double value);

    /**
     * Wait up to timeoutMs for a frame and detect on it. Returns the packed result (see {@link
     * StyxResult}), or an empty array on timeout.
     */
    public static native byte[] processFrame(long handle, long timeoutMs);

    /** The next raw JPEG, or an empty array on timeout. */
    public static native byte[] rawJpeg(long handle, long timeoutMs);

    /** The next processed JPEG (tag outlines), or an empty array on timeout. */
    public static native byte[] processedJpeg(long handle, long timeoutMs);

    /** CLOCK_BOOTTIME now: the clock frame capture times are on. */
    public static native long nowBoottimeNanos();

    /**
     * Ask for a board detection (a {@link StyxCalibration} board JSON) on the next camera frame; the
     * result rides in that frame's {@link StyxResult#board()}.
     */
    public static native void requestBoard(long handle, String boardJson);

    /**
     * Solve the camera from board views: {@code viewSizes[v]} corners of view {@code v}, taken in
     * order from {@code ids} and {@code corners} (u, v pairs). Returns {@code [fx, fy, cx, cy, skew,
     * rms, distortionCount, converged, distortion..., viewRms...]}; see photon-styx/src/lib.rs.
     */
    public static native double[] calibrate(
            String boardJson, String lensModel, int[] viewSizes, int[] ids, double[] corners);
}
