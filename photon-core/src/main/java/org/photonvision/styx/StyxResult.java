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
 * One frame's detections, decoded from the little-endian layout in photon-styx/src/pack.rs (version
 * 4). Poses are {@code [x, y, z, qw, qx, qy, qz]}; a pose with NaN is absent.
 */
public record StyxResult(
        long sequence,
        long captureNanos,
        long processingNanos,
        long[] stages,
        int width,
        int height,
        boolean multiValid,
        int multiInlierTags,
        double multiRmsPx,
        double multiAmbiguity,
        double[] multiPose,
        List<Target> targets,
        Board board) {
    public static final int VERSION = 4;

    /** Native stages in the record (photon-styx pack::STAGES). */
    public static final int NATIVE_STAGES = 8;

    public static final int STAGE_SENSOR = 0;
    public static final int STAGE_DEQUEUED = 1;
    public static final int STAGE_ISP_DONE = 2;
    public static final int STAGE_QUEUED = 3;
    public static final int STAGE_TAKEN = 4;
    public static final int STAGE_DETECT_START = 5;
    public static final int STAGE_DETECT_END = 6;
    public static final int STAGE_PACKED = 7;

    /** The JNI call returned the record to Java. */
    public static final int STAGE_JAVA_RECEIVED = 8;

    /** The pipeline result reached the result consumers (NetworkTables published it). */
    public static final int STAGE_PUBLISHED = 9;

    public static final int STAGES = 10;
    static final int HEADER_BYTES = 4 + 8 + 8 + 8 + 4 + 4 + 4 + NATIVE_STAGES * 8;
    static final int MULTI_BYTES = 4 + 4 + 8 + 8 + 7 * 8;
    static final int TARGET_BYTES = 4 * 4 + 8 * 8 + 7 * 8 + 7 * 8 + 3 * 8 + 2 * 8;

    /**
     * One detected tag. {@code alt} is null when the tag has no alternative pose. {@code center} is
     * the tag centre undistorted into pinhole intrinsics (NaN when unknown).
     */
    public record Target(
            int id,
            int hamming,
            boolean usedInMulti,
            double[] corners,
            double[] best,
            double[] alt,
            double ambiguity,
            double rmsBestPx,
            double rmsAltPx,
            double[] center) {}

    /**
     * A board detection asked for on this frame. {@code corners} is flat (u, v per corner) and {@code
     * ids} the corner ids in the same order; both are empty when the board was not seen.
     */
    public record Board(boolean found, int[] ids, double[] corners) {}

    public static StyxResult decode(byte[] bytes) {
        ByteBuffer b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int version = b.getInt();
        if (version != VERSION) {
            throw new IllegalStateException(
                    "photonstyx result version " + version + ", expected " + VERSION);
        }
        long sequence = b.getLong();
        long capture = b.getLong();
        long processing = b.getLong();
        int width = b.getInt();
        int height = b.getInt();
        int count = b.getInt();
        long[] stages = new long[STAGES];
        for (int i = 0; i < NATIVE_STAGES; i++) stages[i] = b.getLong();
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
            double[] center = doubles(b, 2);
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
                            rmsAlt,
                            center));
        }
        Board board = null;
        if (b.getInt() != 0) {
            boolean found = b.getInt() != 0;
            int corners = b.getInt();
            int[] ids = new int[corners];
            double[] uv = new double[2 * corners];
            for (int i = 0; i < corners; i++) {
                ids[i] = b.getInt();
                uv[2 * i] = b.getDouble();
                uv[2 * i + 1] = b.getDouble();
            }
            board = new Board(found, ids, uv);
        } else {
            b.getInt();
            b.getInt();
        }
        return new StyxResult(
                sequence,
                capture,
                processing,
                stages,
                width,
                height,
                multiValid,
                inliers,
                rms,
                ambiguity,
                multiPose,
                List.copyOf(targets),
                board);
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
