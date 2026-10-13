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

package org.photonvision.common.hardware.gpio;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.ShellExec;

public class CustomAdapter {
    private static final Logger logger = new Logger(CustomAdapter.class, LogGroup.General);
    private static final ThreadLocal<ShellExec> runCommand =
            ThreadLocal.withInitial(() -> new ShellExec(true, true));

    protected final String getGPIOCommand;
    protected final String setGPIOCommand;
    protected final String setPWMCommand;
    protected final String setPWMFrequencyCommand;
    protected final String releaseGPIOCommand;

    // The last command run for each pin. Status LEDs are set many times a second with the same
    // value, and each set is a bash process; a repeat of the last command changes nothing.
    private final Map<Integer, String> lastWrite = new ConcurrentHashMap<>();

    public CustomAdapter(
            String getGPIOCommand,
            String setGPIOCommand,
            String setPWMCommand,
            String setPWMFrequencyCommand,
            String releaseGPIOCommand) {
        this.getGPIOCommand = getGPIOCommand;
        this.setGPIOCommand = setGPIOCommand;
        this.setPWMCommand = setPWMCommand;
        this.setPWMFrequencyCommand = setPWMFrequencyCommand;
        this.releaseGPIOCommand = releaseGPIOCommand;
    }

    protected static String execute(String command) {
        try {
            runCommand.get().executeBashCommand(command);
        } catch (Exception e) {
            logger.error("Exception caught running GPIO command \"" + command + "\"", e);
            return "";
        }
        return runCommand.get().getOutput();
    }

    public boolean getGPIO(int gpio) {
        return Boolean.parseBoolean(
                execute(getGPIOCommand.replace("{p}", Integer.toString(gpio))).trim());
    }

    public void setGPIO(int gpio, boolean state) {
        write(
                gpio,
                setGPIOCommand
                        .replace("{p}", Integer.toString(gpio))
                        .replace("{s}", Boolean.toString(state)));
    }

    public void setPWM(int gpio, double value) {
        write(
                gpio,
                setPWMCommand
                        .replace("{p}", Integer.toString(gpio))
                        .replace("{v}", Double.toString(value)));
    }

    public void setPwmFrequency(int gpio, int frequency) {
        execute(
                setPWMFrequencyCommand
                        .replace("{p}", Integer.toString(gpio))
                        .replace("{f}", Integer.toString(frequency)));
    }

    public void releaseGPIO(int gpio) {
        lastWrite.remove(gpio);
        execute(releaseGPIOCommand.replace("{p}", Integer.toString(gpio)));
    }

    /** Run a pin's set command, unless it is the same command as the last one for that pin. */
    private void write(int gpio, String command) {
        if (command.equals(lastWrite.put(gpio, command))) return;
        execute(command);
    }
}
