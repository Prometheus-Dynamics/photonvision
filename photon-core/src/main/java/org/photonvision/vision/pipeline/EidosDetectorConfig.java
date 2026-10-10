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
package org.photonvision.vision.pipeline;

import java.util.Locale;
import org.photonvision.common.configuration.ConfigProvider;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.CameraLensModel;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.wpilib.fields.Field;
import org.wpilib.fields.FieldTag;
import org.wpilib.math.geometry.Pose3d;

/** Builds the JSON that {@code StyxJNI.configure} takes (photon-styx/src/config.rs). */
final class EidosDetectorConfig {
    private EidosDetectorConfig() {}

    /** Tag edge length in metres, as PhotonVision's AprilTag pipeline assumes it. */
    static double tagSizeMeters(EidosAprilTagPipelineSettings settings) {
        return settings.tagFamily == org.photonvision.vision.apriltag.AprilTagFamily.kTag16h5
                ? 0.1524 // 6 in, 2023 tags
                : 0.165_1; // 6.5 in, 2024 tags
    }

    static String json(EidosAprilTagPipelineSettings settings, FrameStaticProperties props) {
        StringBuilder sb = new StringBuilder(512);
        sb.append('{');
        sb.append("\"family\":\"")
                .append(settings.tagFamily == org.photonvision.vision.apriltag.AprilTagFamily.kTag16h5
                        ? "tag16h5" : "tag36h11")
                .append('"');
        sb.append(",\"tagSizeM\":").append(num(tagSizeMeters(settings)));
        sb.append(",\"multiTag\":").append(settings.doMultiTarget);
        sb.append(",\"fullSearchEvery\":").append(Math.max(1, settings.fullSearchEvery));
        sb.append(",\"processedStream\":").append(settings.processedStream);
        CameraCalibrationCoefficients cal = props == null ? null : props.cameraCalibration;
        if (cal != null) {
            double[] k = cal.getIntrinsicsArr(); // row-major 3x3: fx 0 cx / 0 fy cy / 0 0 1
            double[] d = cal.getDistCoeffsArr();
            sb.append(",\"calibration\":{");
            sb.append("\"fx\":").append(num(k[0]));
            sb.append(",\"fy\":").append(num(k[4]));
            sb.append(",\"cx\":").append(num(k[2]));
            sb.append(",\"cy\":").append(num(k[5]));
            sb.append(",\"skew\":").append(num(k[1]));
            boolean fisheye = cal.lensmodel == CameraLensModel.LENSMODEL_OPENCV_FISHEYE;
            // OpenCV order: k1 k2 p1 p2 k3 k4 k5 k6. A fisheye has k1..k4 (Kannala-Brandt).
            String[] names =
                    fisheye
                            ? new String[] {"k1", "k2", "k3", "k4"}
                            : new String[] {"k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"};
            for (int i = 0; i < names.length && i < d.length; i++) {
                sb.append(",\"").append(names[i]).append("\":").append(num(d[i]));
            }
            sb.append(",\"fisheye\":").append(fisheye);
            sb.append('}');
        }
        if (settings.doMultiTarget) {
            sb.append(",\"fieldTags\":[");
            boolean first = true;
            Field field = ConfigProvider.getInstance().getConfig().getFieldLayout();
            if (field != null && field.getTags() != null) {
                for (FieldTag tag : field.getTags()) {
                    Pose3d pose = tag.getPose();
                    if (pose == null) continue;
                    if (!first) sb.append(',');
                    first = false;
                    sb.append("{\"id\":").append(tag.getID())
                            .append(",\"x\":").append(num(pose.getX()))
                            .append(",\"y\":").append(num(pose.getY()))
                            .append(",\"z\":").append(num(pose.getZ()))
                            .append(",\"qw\":").append(num(pose.getRotation().getQuaternion().getW()))
                            .append(",\"qx\":").append(num(pose.getRotation().getQuaternion().getX()))
                            .append(",\"qy\":").append(num(pose.getRotation().getQuaternion().getY()))
                            .append(",\"qz\":").append(num(pose.getRotation().getQuaternion().getZ()))
                            .append('}');
                }
            }
            sb.append(']');
        }
        sb.append('}');
        return sb.toString();
    }

    private static String num(double v) {
        // Locale-independent, and no NaN/Infinity in JSON.
        if (!Double.isFinite(v)) return "0";
        return String.format(Locale.ROOT, "%.12g", v);
    }
}
