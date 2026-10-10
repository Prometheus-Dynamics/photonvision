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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * One frame's detections, decoded from the little-endian layout in photon-styx/src/pack.rs
 * (version 1). Poses are {@code [x, y, z, qw, qx, qy, qz]}; a pose with NaN is absent.
 */
public record StyxResult(
        long sequence,
        long captureNanos,
        long processingNanos,
        int width,
        int height,
        boolean multiValid,
        int multiInlierTags,
        double multiRmsPx,
        double multiAmbiguity,
        double[] multiPose,
        List<Target> targets) {
    public static final int VERSION = 1;
    static final int HEADER_BYTES = 4 + 8 + 8 + 8 + 4 + 4 + 4;
    static final int MULTI_BYTES = 4 + 4 + 8 + 8 + 7 * 8;
    static final int TARGET_BYTES = 4 * 4 + 8 * 8 + 7 * 8 + 7 * 8 + 3 * 8;

    /** One detected tag. {@code alt} is null when the tag has no alternative pose. */
    public record Target(
            int id,
            int hamming,
            boolean usedInMulti,
            double[] corners,
            double[] best,
            double[] alt,
            double ambiguity,
            double rmsBestPx,
            double rmsAltPx) {}

    public static StyxResult decode(byte[] bytes) {
        ByteBuffer b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int version = b.getInt();
        if (version != VERSION) {
            throw new IllegalStateException("photonstyx result version " + version + ", expected " + VERSION);
        }
        long sequence = b.getLong();
        long capture = b.getLong();
        long processing = b.getLong();
        int width = b.getInt();
        int height = b.getInt();
        int count = b.getInt();
        boolean multiValid = b.getInt() != 0;
        int inliers = b.getInt();
        double rms = b.getDouble();
        double ambiguity = b.getDouble();
        double[] multiPose = doubles(b, 7);
        List<Target> targets = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            int id = b.getInt();
            int hamming = b.getInt();
            boolean used = b.getInt() != 0;
            boolean hasAlt = b.getInt() != 0;
            double[] corners = doubles(b, 8);
            double[] best = doubles(b, 7);
            double[] alt = doubles(b, 7);
            double tagAmbiguity = b.getDouble();
            double rmsBest = b.getDouble();
            double rmsAlt = b.getDouble();
            targets.add(
                    new Target(
                            id,
                            hamming,
                            used,
                            corners,
                            best,
                            hasAlt ? alt : null,
                            tagAmbiguity,
                            rmsBest,
                            rmsAlt));
        }
        return new StyxResult(
                sequence,
                capture,
                processing,
                width,
                height,
                multiValid,
                inliers,
                rms,
                ambiguity,
                multiPose,
                List.copyOf(targets));
    }

    private static double[] doubles(ByteBuffer b, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = b.getDouble();
        return out;
    }

    /** True when the pose is present (not NaN). */
    public static boolean hasPose(double[] pose) {
        return pose != null && pose.length == 7 && !Double.isNaN(pose[0]);
    }
}
