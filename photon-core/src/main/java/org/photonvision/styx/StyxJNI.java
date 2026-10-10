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
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.jni.CombinedRuntimeLoader;

/**
 * JNI into libphotonstyx.so (photon-styx/ in the PhotonVision fork): Styx capture and Eidos AprilTag
 * detection for the experimental Styx backend. Handles are opaque positive longs. Only the Styx
 * backend loads this library (see {@link StyxBackend}).
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

    /**
     * Load libphotonstyx: from {@code -Dphotonvision.styx.library=/path/libphotonstyx.so} when set,
     * otherwise from the jar's native resources.
     */
    public static synchronized void load() throws IOException {
        if (loaded) return;
        String override = System.getProperty("photonvision.styx.library");
        if (override != null && !override.isBlank()) {
            System.load(override);
        } else {
            CombinedRuntimeLoader.loadLibraries(StyxJNI.class, "photonstyx");
        }
        loaded = true;
        logger.info("Loaded libphotonstyx");
    }

    public static synchronized boolean isLoaded() {
        return loaded;
    }

    /** Camera names Styx finds (the names {@link #open} takes). */
    public static native String[] listCameras();

    /** Open a camera at exactly width x height and fps. Returns a handle. */
    public static native long open(String camera, int width, int height, int fps);

    public static native void close(long handle);

    /** Pipeline configuration as JSON (see photon-styx/src/config.rs). */
    public static native void configure(long handle, String json);

    /** Apply a standard control (CONTROL_*); returns the value in effect (NaN if unknown). */
    public static native double setControl(long handle, int kind, double value);

    /**
     * Wait up to timeoutMs for a frame and detect on it. Returns the packed result (see
     * {@link StyxResult}), or an empty array on timeout.
     */
    public static native byte[] processFrame(long handle, long timeoutMs);

    /** The next raw JPEG, or an empty array on timeout. */
    public static native byte[] rawJpeg(long handle, long timeoutMs);

    /** The next processed JPEG (tag outlines), or an empty array on timeout. */
    public static native byte[] processedJpeg(long handle, long timeoutMs);

    /** CLOCK_BOOTTIME now: the clock frame capture times are on. */
    public static native long nowBoottimeNanos();
}
