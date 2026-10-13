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
import static org.junit.jupiter.api.Assertions.assertNull;

import java.util.List;
import org.junit.jupiter.api.Test;

class StyxCamerasTest {
    private static final String STYX = "ov9782 (kernel driver i2c 10-0060)";
    private static final String LIBCAMERA = "/base/axi/pcie@1000120000/rp1/i2c@88000/ov9782@60";

    @Test
    void exactNameWins() {
        assertEquals(STYX, StyxCameras.resolve(STYX, List.of(STYX)));
    }

    @Test
    void libcameraPathMatchesTheSameSensorByAddress() {
        assertEquals(STYX, StyxCameras.resolve(LIBCAMERA, List.of("imx219 (kernel driver i2c 10-0010)", STYX)));
    }

    @Test
    void otherSensorsOrAddressesDoNotMatch() {
        assertNull(StyxCameras.resolve(LIBCAMERA, List.of("imx219 (kernel driver i2c 10-0010)")));
        assertNull(StyxCameras.resolve("/base/axi/x/i2c@88000/ov9782@10", List.of(STYX)));
    }

    @Test
    void noNamesNoMatch() {
        assertNull(StyxCameras.resolve(LIBCAMERA, List.of()));
    }
}
