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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;

/**
 * MJPEG stream for the Styx backend, on the same ports and path (/stream.mjpg) as cscore's
 * MjpegServer. One thread takes JPEGs from the camera session; each client gets the HTTP headers
 * at once (flushed, so a browser or curl sees the stream open) and then the latest frame as it
 * arrives. Raw sockets, not com.sun.net.httpserver: that one does not send the headers until the
 * first body bytes, which left clients waiting on a stream that had no frames yet.
 */
public final class StyxMjpegServer implements AutoCloseable {
    private static final Logger logger = new Logger(StyxMjpegServer.class, LogGroup.Camera);
    private static final String BOUNDARY = "photonstyxframe";
    private static final String HEADERS =
            "HTTP/1.1 200 OK\r\n"
                    + "Content-Type: multipart/x-mixed-replace; boundary=" + BOUNDARY + "\r\n"
                    + "Cache-Control: no-cache\r\n"
                    // The UI loads streams with crossorigin="anonymous" (from port 5800),
                    // so the browser needs CORS, as cscore's MJPEG server sends.
                    + "Access-Control-Allow-Origin: *\r\n"
                    + "Connection: close\r\n\r\n";

    /** Supplies the next JPEG, waiting up to timeoutMs; an empty array means none this time. */
    @FunctionalInterface
    public interface JpegSource {
        byte[] next(long timeoutMs);
    }

    private final String name;
    private final ServerSocket server;
    private final Object lock = new Object();
    private byte[] latest = new byte[0];
    private long sequence = 0;
    private volatile boolean running = true;
    private final Thread acceptor;
    private final Thread feeder;

    public StyxMjpegServer(String name, int port, JpegSource source) throws IOException {
        this.name = name;
        this.server = new ServerSocket(port);
        this.server.setReuseAddress(true);
        this.acceptor = new Thread(this::accept, name + "-accept");
        this.acceptor.setDaemon(true);
        this.feeder = new Thread(() -> feed(source), name + "-feeder");
        this.feeder.setDaemon(true);
    }

    public void start() {
        feeder.start();
        acceptor.start();
        logger.info("Serving " + name + " on port " + server.getLocalPort());
    }

    private void accept() {
        while (running) {
            try {
                Socket client = server.accept();
                Thread t = new Thread(() -> serve(client), name + "-client");
                t.setDaemon(true);
                t.start();
            } catch (IOException e) {
                if (running) logger.debug(name + ": accept failed: " + e.getMessage());
            }
        }
    }

    private void feed(JpegSource source) {
        while (running) {
            byte[] jpeg;
            try {
                jpeg = source.next(500);
            } catch (RuntimeException e) {
                logger.error(name + ": frame source failed", e);
                jpeg = new byte[0];
            }
            if (jpeg == null || jpeg.length == 0) {
                // No session or no frame yet: back off instead of spinning.
                sleep(20);
                continue;
            }
            synchronized (lock) {
                latest = jpeg;
                sequence++;
                lock.notifyAll();
            }
        }
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private void serve(Socket client) {
        try (client) {
            client.setSoTimeout(5000);
            // Read and ignore the request up to its blank line.
            BufferedReader in =
                    new BufferedReader(new InputStreamReader(client.getInputStream(), StandardCharsets.US_ASCII));
            String line;
            do {
                line = in.readLine();
            } while (line != null && !line.isEmpty());
            OutputStream out = client.getOutputStream();
            out.write(HEADERS.getBytes(StandardCharsets.US_ASCII));
            out.flush();
            client.setSoTimeout(0);
            long seen = -1;
            while (running) {
                byte[] jpeg;
                synchronized (lock) {
                    while (running && sequence == seen) {
                        lock.wait(1000);
                    }
                    seen = sequence;
                    jpeg = latest;
                }
                out.write(("--" + BOUNDARY + "\r\nContent-Type: image/jpeg\r\nContent-Length: " + jpeg.length + "\r\n\r\n")
                        .getBytes(StandardCharsets.US_ASCII));
                out.write(jpeg);
                out.write("\r\n".getBytes(StandardCharsets.US_ASCII));
                out.flush();
            }
        } catch (IOException e) {
            // The client went away.
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public void close() {
        running = false;
        synchronized (lock) {
            lock.notifyAll();
        }
        try {
            server.close();
        } catch (IOException e) {
            // Already closed.
        }
    }
}
