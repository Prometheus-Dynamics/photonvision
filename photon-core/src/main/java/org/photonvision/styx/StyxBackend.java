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

/**
 * Whether this process runs the experimental Styx backend (Styx capture, Eidos AprilTags, no
 * libcamera). Selected with {@code -Dphotonvision.backend=styx} or the environment variable
 * {@code PHOTONVISION_BACKEND=styx}. The normal build leaves it off.
 */
public final class StyxBackend {
    private StyxBackend() {}

    public static boolean enabled() {
        String property = System.getProperty("photonvision.backend");
        if (property != null) return "styx".equalsIgnoreCase(property.trim());
        return "styx".equalsIgnoreCase(System.getenv("PHOTONVISION_BACKEND"));
    }
}
