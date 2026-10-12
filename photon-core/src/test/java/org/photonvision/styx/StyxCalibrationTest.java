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
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import org.photonvision.vision.calibration.CameraLensModel;
import org.photonvision.vision.pipeline.UICalibrationData;

class StyxCalibrationTest {
    @Test
    void chessboardJsonCarriesSquaresAndSquareSize() {
        assertEquals(
                "{\"kind\":\"chessboard\",\"squaresX\":9,\"squaresY\":7,\"squareM\":0.0254}",
                StyxCalibration.boardJson(
                        UICalibrationData.BoardType.CHESSBOARD,
                        9,
                        7,
                        0.0254,
                        0.0,
                        UICalibrationData.TagFamily.Dict_4X4_1000,
                        false));
    }

    @Test
    void charucoJsonCarriesMarkersDictionaryAndPattern() {
        assertEquals(
                "{\"kind\":\"charuco\",\"squaresX\":6,\"squaresY\":5,\"squareM\":0.04,"
                        + "\"markerM\":0.03,\"dictionary\":\"5x5_1000\",\"legacy\":true}",
                StyxCalibration.boardJson(
                        UICalibrationData.BoardType.CHARUCOBOARD,
                        6,
                        5,
                        0.04,
                        0.03,
                        UICalibrationData.TagFamily.Dict_5X5_1000,
                        true));
    }

    @Test
    void lensModelsMapToTheSolvedModelsOnly() {
        assertEquals("pinhole", StyxCalibration.lensName(CameraLensModel.LENSMODEL_OPENCV));
        assertEquals("fisheye", StyxCalibration.lensName(CameraLensModel.LENSMODEL_OPENCV_FISHEYE));
        assertThrows(
                IllegalArgumentException.class,
                () -> StyxCalibration.lensName(CameraLensModel.LENSMODEL_SPLINED_STERIOGRAPHIC));
    }

    @Test
    void nonFiniteLengthsAreRefused() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        StyxCalibration.boardJson(
                                UICalibrationData.BoardType.CHESSBOARD,
                                9,
                                7,
                                Double.NaN,
                                0.0,
                                UICalibrationData.TagFamily.Dict_4X4_1000,
                                false));
    }
}
