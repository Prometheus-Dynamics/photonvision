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

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Maps a configured CSI camera path to the name Styx gives the same sensor. A camera configured on
 * a libcamera image keeps libcamera's device path (e.g. {@code /base/.../i2c@88000/ov9782@60}); on
 * Styx the sensor is named by its kernel driver and I2C address ({@code ov9782 (kernel driver i2c
 * 10-0060)}). The sensor model and the I2C address (the {@code @60} is hex 0x60, shown as 0060)
 * identify it on both.
 */
public final class StyxCameras {
    private static final Pattern LIBCAMERA_PATH = Pattern.compile("/([A-Za-z0-9]+)@([0-9a-fA-F]+)$");
    private static final long CACHE_NANOS = 2_000_000_000L;
    private static List<String> cachedNames = List.of();
    private static long cachedAt = Long.MIN_VALUE / 2;

    private StyxCameras() {}

    /** The Styx camera name for a configured path, or null when no Styx camera matches. */
    public static synchronized String resolve(String configuredPath) {
        if (configuredPath == null) return null;
        return resolve(configuredPath, names());
    }

    /** The names Styx lists, probed at most every two seconds. */
    private static List<String> names() {
        long now = System.nanoTime();
        if (now - cachedAt > CACHE_NANOS) {
            cachedAt = now;
            try {
                StyxJNI.load();
                cachedNames = List.of(StyxJNI.listCameras());
            } catch (IOException | RuntimeException | UnsatisfiedLinkError e) {
                cachedNames = List.of();
            }
        }
        return cachedNames;
    }

    /** The matching rule, on given names (exposed for tests). */
    static String resolve(String configuredPath, List<String> names) {
        if (names.contains(configuredPath)) return configuredPath;
        Matcher m = LIBCAMERA_PATH.matcher(configuredPath);
        if (!m.find()) return null;
        String sensor = m.group(1).toLowerCase(Locale.ROOT);
        String address = String.format(Locale.ROOT, "%04x", Integer.parseInt(m.group(2), 16));
        for (String name : names) {
            String lower = name.toLowerCase(Locale.ROOT);
            if (lower.contains(sensor) && lower.contains("-" + address)) return name;
        }
        return null;
    }
}
