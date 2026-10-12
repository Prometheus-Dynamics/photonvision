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

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.targeting.TargetCorner;
import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.geometry.Transform3d;
import edu.wpi.first.math.geometry.Translation3d;

/** Eidos poses and corners against the conventions of PhotonVision's own AprilTag pipeline. */
class StyxTargetMathTest {
    private static double[] pose(Transform3d t) {
        var q = t.getRotation().getQuaternion();
        return new double[] {t.getX(), t.getY(), t.getZ(), q.getW(), q.getX(), q.getY(), q.getZ()};
    }

    @Test
    void aTagFacingTheCameraIsAheadAndTurnedAround() {
        // A tag 2 m straight ahead, facing the camera: in the AprilTag (and Eidos) convention the
        // tag frame is x right, y down, z into the tag, so camera-from-tag is the identity rotation.
        double[] eidos = {0, 0, 2, 1, 0, 0, 0};
        Transform3d photon = StyxTargetMath.opticalPoseToPhoton(eidos);
        assertEquals(2.0, photon.getX(), 1e-9);
        assertEquals(0.0, photon.getY(), 1e-9);
        assertEquals(0.0, photon.getZ(), 1e-9);
        // PhotonVision reports such a tag turned 180 degrees about up (its x out of its face,
        // towards the camera), as its AprilTag pipeline does.
        assertEquals(Math.PI, Math.abs(photon.getRotation().getZ()), 1e-9);
        assertEquals(0.0, photon.getRotation().getX(), 1e-9);
        assertEquals(0.0, photon.getRotation().getY(), 1e-9);
    }

    @Test
    void matchesPhotonVisionsAprilTagConversion() {
        // Any camera-from-tag pose from the AprilTag estimator: PhotonVision's AprilTag pipeline
        // applies convertApriltagtoOpenCV (TrackedTarget) then convertOpenCVtoPhotonTransform.
        var apriltag =
                new Transform3d(new Translation3d(0.4, -0.2, 3.1), new Rotation3d(0.3, -0.5, 0.2));
        Transform3d expected =
                MathUtils.convertOpenCVtoPhotonTransform(MathUtils.convertApriltagtoOpenCV(apriltag));
        Transform3d actual = StyxTargetMath.opticalPoseToPhoton(pose(apriltag));
        assertEquals(expected.getX(), actual.getX(), 1e-9);
        assertEquals(expected.getY(), actual.getY(), 1e-9);
        assertEquals(expected.getZ(), actual.getZ(), 1e-9);
        assertEquals(
                1.0,
                Math.abs(expected.getRotation().getQuaternion().dot(actual.getRotation().getQuaternion())),
                1e-9);
        // A tag 3.1 m ahead, 0.4 m right and 0.2 m up of the axis: NWU x forward, y left, z up.
        assertEquals(3.1, actual.getX(), 1e-9);
        assertEquals(-0.4, actual.getY(), 1e-9);
        assertEquals(0.2, actual.getZ(), 1e-9);
    }

    @Test
    void cornersAreInAprilTagOrder() {
        // Eidos marker order: top-left, top-right, bottom-right, bottom-left (of an upright tag).
        double[] eidos = {10, 10, 20, 10, 20, 20, 10, 20};
        List<TargetCorner> c = StyxTargetMath.photonCorners(eidos);
        // AprilTag order: bottom-left, bottom-right, top-right, top-left.
        assertEquals(new TargetCorner(10, 20), c.get(0));
        assertEquals(new TargetCorner(20, 20), c.get(1));
        assertEquals(new TargetCorner(20, 10), c.get(2));
        assertEquals(new TargetCorner(10, 10), c.get(3));
    }
}
