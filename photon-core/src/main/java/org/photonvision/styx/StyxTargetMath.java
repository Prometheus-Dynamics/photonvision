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

import java.util.ArrayList;
import java.util.List;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.styx.StyxResult.Target;
import org.photonvision.targeting.MultiTargetPNPResult;
import org.photonvision.targeting.PhotonTrackedTarget;
import org.photonvision.targeting.PnpResult;
import org.photonvision.targeting.TargetCorner;
import edu.wpi.first.math.geometry.Quaternion;
import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.geometry.Transform3d;
import edu.wpi.first.math.geometry.Translation3d;

/**
 * Turns Styx/Eidos detections into PhotonVision's result types, with the same formulas as
 * {@code TrackedTarget} (AprilTag path) and {@code TargetCalculations}, but without OpenCV.
 *
 * <p>The centre used for yaw and pitch is undistorted by Eidos (PhotonVision does the same for
 * calibrated cameras); the area comes from the distorted corners.
 */
public final class StyxTargetMath {
    private StyxTargetMath() {}

    /** Camera intrinsics and frame size the target values need. */
    public record Intrinsics(double fx, double fy, double cx, double cy, double imageArea) {}

    /**
     * An Eidos camera-from-tag pose {@code [x,y,z,qw,qx,qy,qz]} as PhotonVision's camera-to-target.
     *
     * <p>Eidos's camera frame is optical (EDN) and its tag frame is the AprilTag library's (x right,
     * y down, z into the tag), so the pose is what WPILib's AprilTagPoseEstimator returns for the
     * same corners. It goes through the same two steps as PhotonVision's AprilTag pipeline
     * ({@code TrackedTarget}, then {@code AprilTagPipeline}): the tag's z is turned to point out of
     * the tag, then EDN becomes NWU.
     */
    public static Transform3d opticalPoseToPhoton(double[] pose) {
        if (!StyxResult.hasPose(pose)) return new Transform3d();
        Transform3d cameraToTarget =
                new Transform3d(
                        new Translation3d(pose[0], pose[1], pose[2]),
                        new Rotation3d(new Quaternion(pose[3], pose[4], pose[5], pose[6])));
        return MathUtils.convertOpenCVtoPhotonTransform(MathUtils.convertApriltagtoOpenCV(cameraToTarget));
    }

    /**
     * Eidos reports corners in marker order (top-left, top-right, bottom-right, bottom-left in the
     * tag's own frame); PhotonVision's fiducial corners are the AprilTag library's order
     * (bottom-left, bottom-right, top-right, top-left: counter-clockwise from bottom-left). Index
     * {@code i} of PhotonVision's list is Eidos corner {@code 3 - i}.
     */
    public static List<TargetCorner> photonCorners(double[] eidosCorners) {
        List<TargetCorner> corners = new ArrayList<>(4);
        for (int i = 0; i < 4; i++) {
            int e = 3 - i;
            corners.add(new TargetCorner(eidosCorners[2 * e], eidosCorners[2 * e + 1]));
        }
        return corners;
    }

    /** Field-frame WPILib pose {@code [x,y,z,qw,qx,qy,qz]} as a Transform3d from the origin. */
    public static Transform3d fieldPose(double[] pose) {
        if (!StyxResult.hasPose(pose)) return new Transform3d();
        return new Transform3d(
                new Translation3d(pose[0], pose[1], pose[2]),
                new Rotation3d(new Quaternion(pose[3], pose[4], pose[5], pose[6])));
    }

    public static PhotonTrackedTarget toPhotonTarget(Target t, Intrinsics k) {
        return toPhotonTarget(t, k, true);
    }

    /** {@code withPose} false (2D mode) leaves the poses empty, as PhotonVision's 2D pipelines do. */
    public static PhotonTrackedTarget toPhotonTarget(Target t, Intrinsics k, boolean withPose) {
        double[] c = t.corners();
        // The undistorted centre from Eidos when it has one; else the mean of the corners.
        double centerX = Double.isFinite(t.center()[0]) ? t.center()[0] : (c[0] + c[2] + c[4] + c[6]) / 4.0;
        double centerY = Double.isFinite(t.center()[1]) ? t.center()[1] : (c[1] + c[3] + c[5] + c[7]) / 4.0;
        double yaw = Math.atan((centerX - k.cx()) / k.fx());
        double pitch = Math.atan((k.cy() - centerY) / (k.fy() / Math.cos(yaw)));

        List<TargetCorner> corners = photonCorners(c);
        // Shoelace area of the quad, as a fraction of the image in percent (TrackedTarget's area).
        double area2 = 0;
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            area2 += c[2 * i] * c[2 * j + 1] - c[2 * j] * c[2 * i + 1];
        }
        double area = Math.abs(area2) / 2.0 / k.imageArea() * 100.0;

        Transform3d best = withPose ? opticalPoseToPhoton(t.best()) : new Transform3d();
        Transform3d alt = withPose && t.alt() != null ? opticalPoseToPhoton(t.alt()) : new Transform3d();
        return new PhotonTrackedTarget(
                Math.toDegrees(yaw),
                Math.toDegrees(pitch),
                area,
                0.0, // skew: AprilTags have none in PhotonVision either
                t.id(),
                -1,
                0f,
                best,
                alt,
                withPose && StyxResult.hasPose(t.best()) ? t.ambiguity() : -1,
                corners,
                new ArrayList<>(corners));
    }

    /** Multi-tag result: the camera pose in the field frame and the tags that made it. */
    public static MultiTargetPNPResult toMultiTarget(StyxResult r) {
        Transform3d pose = fieldPose(r.multiPose());
        PnpResult pnp = new PnpResult(pose, pose, r.multiAmbiguity(), r.multiRmsPx(), r.multiRmsPx());
        List<Short> used = new ArrayList<>();
        for (Target t : r.targets()) {
            if (t.usedInMulti()) used.add((short) t.id());
        }
        return new MultiTargetPNPResult(pnp, used);
    }
}
