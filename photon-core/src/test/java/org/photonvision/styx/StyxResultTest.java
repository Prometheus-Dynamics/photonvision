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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;
import org.photonvision.targeting.PhotonTrackedTarget;

class StyxResultTest {
    /** Bytes laid out as photon-styx/src/pack.rs writes them (version 3). */
    private static byte[] sample(boolean withAlt) {
        ByteBuffer b = ByteBuffer.allocate(40 + 64 + 80 + 232).order(ByteOrder.LITTLE_ENDIAN);
        b.putInt(3).putLong(7).putLong(123_456).putLong(900).putInt(1280).putInt(800).putInt(1);
        for (long stage = 1; stage <= 8; stage++) b.putLong(stage * 1000);
        b.putInt(1).putInt(2).putDouble(0.5).putDouble(0.25);
        for (double v : new double[] {1, 2, 3, 1, 0, 0, 0}) b.putDouble(v);
        b.putInt(11).putInt(1).putInt(1).putInt(withAlt ? 1 : 0);
        for (int i = 0; i < 8; i++) b.putDouble(100 + i);
        for (double v : new double[] {0.1, 0.2, 0.3, 1, 0, 0, 0}) b.putDouble(v);
        for (double v : new double[] {0.4, 0.5, 0.6, 1, 0, 0, 0}) b.putDouble(v);
        b.putDouble(0.3).putDouble(0.4).putDouble(0.6);
        b.putDouble(420).putDouble(300);
        return b.array();
    }

    @Test
    void decodesHeaderMultiTagAndTargets() {
        StyxResult r = StyxResult.decode(sample(true));
        assertEquals(7, r.sequence());
        assertEquals(123_456, r.captureNanos());
        assertEquals(1000, r.stages()[StyxResult.STAGE_SENSOR]);
        assertEquals(8000, r.stages()[StyxResult.STAGE_PACKED]);
        assertEquals(0, r.stages()[StyxResult.STAGE_JAVA_RECEIVED]);
        assertEquals(1280, r.width());
        assertTrue(r.multiValid());
        assertEquals(2, r.multiInlierTags());
        assertArrayEquals(new double[] {1, 2, 3, 1, 0, 0, 0}, r.multiPose());
        assertEquals(1, r.targets().size());
        StyxResult.Target t = r.targets().get(0);
        assertEquals(11, t.id());
        assertTrue(t.usedInMulti());
        assertEquals(100, t.corners()[0]);
        assertArrayEquals(new double[] {0.4, 0.5, 0.6, 1, 0, 0, 0}, t.alt());
        assertEquals(0.3, t.ambiguity());
    }

    @Test
    void altPoseIsNullWhenAbsent() {
        assertNull(StyxResult.decode(sample(false)).targets().get(0).alt());
    }

    @Test
    void rejectsOtherVersions() {
        byte[] bytes = sample(true);
        bytes[0] = 9;
        org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, () -> StyxResult.decode(bytes));
    }

    @Test
    void targetMathCentresTheTagAndMeasuresItsArea() {
        // A 100x100 square centred at (420, 300) in a 640x400 image with fx=fy=500, cx=320, cy=200.
        double[] corners = {370, 250, 470, 250, 470, 350, 370, 350};
        StyxResult.Target t =
                new StyxResult.Target(
                        3, 0, false, corners, new double[] {Double.NaN}, null, 0, 0, 0, new double[] {420, 300});
        PhotonTrackedTarget p =
                StyxTargetMath.toPhotonTarget(
                        t, new StyxTargetMath.Intrinsics(500, 500, 320, 200, 640.0 * 400.0));
        double yaw = Math.toDegrees(Math.atan((420 - 320) / 500.0));
        assertEquals(yaw, p.getYaw(), 1e-9);
        // Pitch: positive when the target is above the optical axis (y down in the image).
        assertTrue(p.getPitch() < 0, "target below the axis: y 300 > cy 200");
        assertEquals(100.0 * 100.0 / (640.0 * 400.0) * 100.0, p.getArea(), 1e-9);
        assertEquals(0.0, p.getSkew());
        assertEquals(4, p.getDetectedCorners().size());
        assertEquals(3, p.getFiducialId());
    }

    @Test
    void absentPoseGivesIdentity() {
        StyxResult.Target t =
                new StyxResult.Target(
                        1, 0, false, new double[8], new double[] {Double.NaN, 0, 0, 0, 0, 0, 0}, null, 0, 0, 0, new double[] {Double.NaN, Double.NaN});
        PhotonTrackedTarget p =
                StyxTargetMath.toPhotonTarget(t, new StyxTargetMath.Intrinsics(1, 1, 0, 0, 1));
        assertEquals(0.0, p.getBestCameraToTarget().getX());
    }
}
