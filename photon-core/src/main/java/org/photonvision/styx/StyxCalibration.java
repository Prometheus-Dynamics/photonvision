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
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.photonvision.vision.calibration.BoardObservation;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.CameraLensModel;
import org.photonvision.vision.calibration.JsonMatOfDouble;
import org.photonvision.vision.pipeline.UICalibrationData;

/**
 * Board calibration on the Styx backend, without OpenCV Mats. The board is found in the camera
 * session ({@link StyxJNI#requestBoard}, the result rides in {@link StyxResult#board()}) and the
 * camera is solved by Eidos ({@link StyxJNI#calibrate}, photon-styx/src/calibrate.rs).
 * PhotonVision's calibration UI, its snapshot flow and the saved coefficients are unchanged.
 */
public final class StyxCalibration {
    private StyxCalibration() {}

    /** One snapshot: corner ids (chessboard or ChArUco) and their pixels, u and v per corner. */
    public record View(int[] ids, double[] corners) {
        public static View of(StyxResult.Board board) {
            return new View(board.ids(), board.corners());
        }
    }

    /**
     * The solved camera. {@code distortion} is in OpenCV's order: k1 k2 p1 p2 k3 (pinhole) or k1 k2
     * k3 k4 (fisheye). {@code viewRms} is per snapshot, in order.
     */
    public record Solved(
            double fx,
            double fy,
            double cx,
            double cy,
            double skew,
            double rms,
            double[] distortion,
            double[] viewRms,
            boolean converged) {}

    /**
     * The board description photon-styx/src/calibrate.rs reads. {@code squaresX} and {@code squaresY}
     * are PhotonVision's board width and height in squares; the lengths are metres.
     */
    public static String boardJson(
            UICalibrationData.BoardType type,
            int squaresX,
            int squaresY,
            double squareMeters,
            double markerMeters,
            UICalibrationData.TagFamily dictionary,
            boolean legacy) {
        requireFinite("square size", squareMeters);
        StringBuilder sb = new StringBuilder(128);
        sb.append("{\"kind\":\"")
                .append(type == UICalibrationData.BoardType.CHARUCOBOARD ? "charuco" : "chessboard")
                .append("\",\"squaresX\":")
                .append(squaresX)
                .append(",\"squaresY\":")
                .append(squaresY)
                .append(",\"squareM\":")
                .append(squareMeters);
        if (type == UICalibrationData.BoardType.CHARUCOBOARD) {
            requireFinite("marker size", markerMeters);
            sb.append(",\"markerM\":").append(markerMeters);
            sb.append(",\"dictionary\":\"").append(dictionaryName(dictionary)).append('"');
            sb.append(",\"legacy\":").append(legacy);
        }
        return sb.append('}').toString();
    }

    /** PhotonVision's dictionary name, as photon-styx/src/calibrate.rs maps it: {@code 4x4_1000}. */
    static String dictionaryName(UICalibrationData.TagFamily dictionary) {
        return dictionary.name().substring("Dict_".length()).toLowerCase(Locale.ROOT);
    }

    /** The lens model name photon-styx/src/calibrate.rs solves: pinhole or fisheye only. */
    public static String lensName(CameraLensModel lens) {
        if (lens == null) return "pinhole";
        return switch (lens) {
            case LENSMODEL_OPENCV -> "pinhole";
            case LENSMODEL_OPENCV_FISHEYE -> "fisheye";
            default ->
                    throw new IllegalArgumentException(
                            "Lens model "
                                    + lens
                                    + " is not supported on the Styx backend (use the OpenCV or the"
                                    + " OpenCV fisheye model)");
        };
    }

    /** Solves the camera over {@code views} (at least three, with at least four corners each). */
    public static Solved solve(String boardJson, CameraLensModel lens, List<View> views) {
        int[] sizes = new int[views.size()];
        int total = 0;
        for (int i = 0; i < sizes.length; i++) {
            sizes[i] = views.get(i).ids().length;
            total += sizes[i];
        }
        int[] ids = new int[total];
        double[] corners = new double[2 * total];
        int at = 0;
        for (View view : views) {
            System.arraycopy(view.ids(), 0, ids, at, view.ids().length);
            System.arraycopy(view.corners(), 0, corners, 2 * at, view.corners().length);
            at += view.ids().length;
        }
        double[] out = StyxJNI.calibrate(boardJson, lensName(lens), sizes, ids, corners);
        int distortionCount = (int) out[6];
        return new Solved(
                out[0],
                out[1],
                out[2],
                out[3],
                out[4],
                out[5],
                Arrays.copyOfRange(out, 8, 8 + distortionCount),
                Arrays.copyOfRange(out, 8 + distortionCount, out.length),
                out[7] != 0);
    }

    /**
     * PhotonVision's coefficients for a solve, as the OpenCV path builds them. The observations carry
     * the snapshots' corners and ids; the per-corner residuals and the board poses are not computed
     * on this backend, so those fields are empty.
     */
    public static CameraCalibrationCoefficients toCoefficients(
            Solved solved,
            List<View> views,
            Size resolution,
            CameraLensModel lens,
            int squaresX,
            int squaresY,
            double squareMeters) {
        JsonMatOfDouble intrinsics =
                new JsonMatOfDouble(
                        3,
                        3,
                        new double[] {
                            solved.fx(), solved.skew(), solved.cx(), 0, solved.fy(), solved.cy(), 0, 0, 1
                        });
        JsonMatOfDouble distortion =
                new JsonMatOfDouble(1, solved.distortion().length, solved.distortion());
        List<BoardObservation> observations = new ArrayList<>(views.size());
        for (View view : views) {
            List<Point> image = new ArrayList<>(view.ids().length);
            for (int i = 0; i < view.ids().length; i++) {
                image.add(new Point(view.corners()[2 * i], view.corners()[2 * i + 1]));
            }
            boolean[] used = new boolean[view.ids().length];
            Arrays.fill(used, true);
            observations.add(new BoardObservation(List.of(), image, List.of(), null, used, null, null));
        }
        return new CameraCalibrationCoefficients(
                resolution,
                intrinsics,
                distortion,
                new double[0],
                observations,
                new Size(squaresX, squaresY),
                squareMeters,
                lens == null ? CameraLensModel.LENSMODEL_OPENCV : lens);
    }

    private static void requireFinite(String name, double value) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException("The " + name + " must be a finite length, got " + value);
        }
    }
}
