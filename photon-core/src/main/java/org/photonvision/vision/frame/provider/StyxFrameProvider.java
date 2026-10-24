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
package org.photonvision.vision.frame.provider;

import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.styx.StyxCameras;
import org.photonvision.styx.StyxJNI;
import org.photonvision.styx.StyxResult;
import org.photonvision.vision.camera.csi.StyxSettables;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameProvider;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.frame.StyxFrame;
import org.photonvision.vision.opencv.ImageRotationMode;
import org.photonvision.vision.pipe.impl.HSVPipe.HSVParams;

/**
 * Frames from a Styx camera. Each {@link #get()} waits for the next camera frame and runs the
 * detector configured on the session; the result rides in a {@link StyxFrame}. No pixels cross into
 * Java, so the threshold, rotation, HSV and copy requests have nothing to do here.
 */
public class StyxFrameProvider extends FrameProvider {
    static final Logger logger = new Logger(StyxFrameProvider.class, LogGroup.Camera);
    private static final long WAIT_MS = 500;

    private final StyxSettables settables;
    private int badFrameCounter = 0;
    private long lastOpenAttempt = Long.MIN_VALUE / 2;
    private static final long OPEN_RETRY_NANOS = 2_000_000_000L;

    public StyxFrameProvider(StyxSettables settables) {
        this.settables = settables;
    }

    @Override
    public String getName() {
        return "StyxFrameProvider";
    }

    @Override
    public Frame get() {
        long handle = settables.handle();
        if (handle == 0) {
            // No session yet: open it (the settables do not, since their properties count as cached),
            // retrying at most every two seconds.
            long now = System.nanoTime();
            if (now - lastOpenAttempt > OPEN_RETRY_NANOS) {
                lastOpenAttempt = now;
                if (checkCameraConnected()) {
                    settables.setVideoMode(settables.getCurrentVideoMode());
                }
            }
            if (settables.handle() == 0) {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                return new Frame();
            }
            handle = settables.handle();
        }
        // The camera is there and streaming: what the UI's "connected" (and its camera controls) read.
        cameraPropertiesCached = true;
        byte[] bytes;
        synchronized (settables.CAMERA_LOCK) {
            if (settables.handle() != handle) return new Frame();
            bytes = StyxJNI.processFrame(handle, WAIT_MS);
        }
        if (bytes.length == 0) {
            badFrameCounter++;
            if (badFrameCounter % 10 == 0) {
                logger.error("No new frame from " + settables.getConfiguration().nickname);
            }
            return new Frame();
        }
        badFrameCounter = 0;

        long received = StyxJNI.nowBoottimeNanos();
        StyxResult result = StyxResult.decode(bytes);
        result.stages()[StyxResult.STAGE_JAVA_RECEIVED] = received;
        ++sequenceID;
        // Styx stamps the start of the frame on the boottime clock; PhotonVision's frame timestamps are
        // on WPILib's clock in nanoseconds (MathUtils.wpiNanoTime, as the libcamera provider uses), so
        // the latency is subtracted from that. NetworkTablesJNI.now() is in microseconds: subtracting
        // nanoseconds from it gave every frame a timestamp seconds in the past.
        long now = MathUtils.wpiNanoTime();
        long timestamp =
                result.captureNanos() > 0
                        ? now - Math.max(0, StyxJNI.nowBoottimeNanos() - result.captureNanos())
                        : now;
        return new StyxFrame(
                sequenceID, handle, result, timestamp, settables.getFrameStaticProperties());
    }

    @Override
    public void requestFrameThresholdType(FrameThresholdType type) {
        // The Eidos pipeline reads the Styx session, not thresholded frames.
    }

    @Override
    public void requestFrameRotation(ImageRotationMode rotationMode) {
        // Rotation is not applied on this path yet.
    }

    @Override
    public void requestHsvSettings(HSVParams params) {}

    @Override
    public void requestFrameCopies(boolean copyInput, boolean copyOutput) {}

    @Override
    public void requestBlockForFrames(boolean blockForFrames) {}

    @Override
    public void release() {
        settables.close();
    }

    @Override
    public boolean isConnected() {
        return checkCameraConnected();
    }

    @Override
    public boolean checkCameraConnected() {
        // A config from a libcamera image keeps libcamera's path; StyxCameras finds the same sensor.
        return StyxCameras.resolve(settables.getConfiguration().matchedCameraInfo.path()) != null;
    }

    @Override
    protected void onCameraConnected() {
        super.onCameraConnected();
        // Opens the session in the current mode (the camera is only opened once it is there).
        settables.setVideoMode(settables.getCurrentVideoMode());
    }
}
