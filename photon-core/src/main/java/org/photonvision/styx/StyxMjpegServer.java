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

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;

/**
 * MJPEG stream for the Styx backend: one thread takes JPEGs from the camera session, and each HTTP
 * client of {@code /stream.mjpg} gets the latest one (multipart/x-mixed-replace), as cscore's
 * MjpegServer serves the other cameras. Same ports as the cscore streams.
 *
 * <p>Frames are only pulled from the session while a client is connected (the session encodes a
 * stream only while it is asked for, at most 30 fps). Each client is sent the newest frame it has
 * not seen; a slow client skips frames and never holds up the others or the camera.
 */
public final class StyxMjpegServer implements AutoCloseable {
    private static final Logger logger = new Logger(StyxMjpegServer.class, LogGroup.Camera);
    private static final String BOUNDARY = "photonstyxframe";

    /** Supplies the next JPEG, waiting up to timeoutMs; an empty array means none this time. */
    @FunctionalInterface
    public interface JpegSource {
        byte[] next(long timeoutMs);
    }

    private final String name;
    private final HttpServer server;
    private final ExecutorService clients;
    private final Object lock = new Object();
    private byte[] latest = new byte[0];
    private long sequence = 0;
    private volatile boolean running = true;
    private final java.util.concurrent.atomic.AtomicInteger clientCount =
            new java.util.concurrent.atomic.AtomicInteger();
    private final Thread feeder;

    public StyxMjpegServer(String name, int port, JpegSource source) throws IOException {
        this.name = name;
        this.server = HttpServer.create(new InetSocketAddress(port), 0);
        this.clients = Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, name + "-client");
            t.setDaemon(true);
            return t;
        });
        server.createContext("/stream.mjpg", exchange -> serve(exchange.getResponseBody(), exchange));
        server.setExecutor(clients);
        feeder = new Thread(() -> feed(source), name + "-feeder");
        feeder.setDaemon(true);
    }

    public void start() {
        feeder.start();
        server.start();
        logger.info("Serving " + name + " on port " + server.getAddress().getPort());
    }

    private void feed(JpegSource source) {
        while (running) {
            try {
                if (clientCount.get() == 0) {
                    synchronized (lock) {
                        lock.wait(250);
                    }
                    continue;
                }
                byte[] jpeg = source.next(500);
                if (jpeg != null && jpeg.length > 0) {
                    synchronized (lock) {
                        latest = jpeg;
                        sequence++;
                        lock.notifyAll();
                    }
                }
            } catch (InterruptedException e) {
                return;
            } catch (RuntimeException e) {
                logger.error(name + ": frame source failed", e);
                try {
                    Thread.sleep(200);
                } catch (InterruptedException ie) {
                    return;
                }
            }
        }
    }

    private void serve(OutputStream out, com.sun.net.httpserver.HttpExchange exchange) throws IOException {
        exchange.getResponseHeaders().add("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY);
        exchange.getResponseHeaders().add("Cache-Control", "no-cache");
        // The UI loads the stream into an <img crossorigin="anonymous"> on another port, which the
        // browser shows only with this header (cscore's MJPEG server sends it too).
        exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        exchange.sendResponseHeaders(200, 0);
        long seen = -1;
        clientCount.incrementAndGet();
        synchronized (lock) {
            lock.notifyAll();
        }
        try {
            while (running) {
                byte[] jpeg;
                synchronized (lock) {
                    while (running && sequence == seen) {
                        try {
                            lock.wait(1000);
                        } catch (InterruptedException e) {
                            return;
                        }
                    }
                    seen = sequence;
                    jpeg = latest;
                }
                if (jpeg.length == 0) continue;
                out.write(("--" + BOUNDARY + "\r\nContent-Type: image/jpeg\r\nContent-Length: " + jpeg.length + "\r\n\r\n")
                        .getBytes(java.nio.charset.StandardCharsets.US_ASCII));
                out.write(jpeg);
                out.write("\r\n".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
                out.flush();
            }
        } catch (IOException e) {
            // The client went away.
        } finally {
            clientCount.decrementAndGet();
            exchange.close();
        }
    }

    @Override
    public void close() {
        running = false;
        synchronized (lock) {
            lock.notifyAll();
        }
        server.stop(0);
        clients.shutdownNow();
    }
}
