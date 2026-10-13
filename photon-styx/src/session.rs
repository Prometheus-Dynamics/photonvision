//! One camera: a Styx grey capture (the kernel-driven sensor's 3A-processed luma, through the
//! PiSP ISP), the Eidos detector, the raw JPEG preview and the processed JPEG. Follows the
//! reference flow in Styx's `pv_capture` (styx-cm5-prep): frames are offered to the preview
//! with `offer_owned`, so a lease is released as soon as the detector has read it.
//!
//! The vision thread only detects: both streams are encoded on threads of their own, at most
//! [`STREAM_FPS`] frames a second, only while someone watches, and from the newest frame only (a
//! slow viewer never holds up capture or detection).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use styx::capture_api::StandardControls;
use styx::codec::jpeg_planar::{JpegBackend, JpegInput, PlanarJpegEncoder};
use styx::core::prelude::Hop;
use styx::prelude::*;
use styx::preview::{Preview, PreviewConfig, PreviewSubscriber};

use crate::calibrate::{self, Board};
use crate::config::PipelineConfig;
use crate::detect::{Detections, Detector, Overlay};
use crate::overlay::{self, Canvas};
use crate::pack::{self, BoardRecord, FrameRecord, STAGES};

/// The raw preview: at most 640x400, 30 fps, JPEG quality 70 (encoded off the vision thread).
const PREVIEW_MAX_SIZE: (u32, u32) = (640, 400);
const PREVIEW_QUALITY: u8 = 70;
/// Most frames a second either stream sends (the link to the viewer is the limit, not the
/// camera: 120 fps of JPEGs swamps it).
pub const STREAM_FPS: f32 = 30.0;
const PROCESSED_QUALITY: u8 = 70;
/// The processed stream is at most this wide (the raw preview's size).
const PROCESSED_MAX_WIDTH: usize = 640;
/// A stream nobody asked for in this long stops being produced.
const WATCH_TIMEOUT_NS: u64 = 2_000_000_000;

/// Control kinds for [`Session::set_control`] (mirrored in `StyxJNI.java`).
pub const CONTROL_EXPOSURE_US: i32 = 1;
pub const CONTROL_GAIN: i32 = 2;
pub const CONTROL_AE: i32 = 3;
pub const CONTROL_EV: i32 = 4;
pub const CONTROL_FPS: i32 = 5;
pub const CONTROL_AWB: i32 = 6;
pub const CONTROL_COLOUR_TEMPERATURE: i32 = 7;
pub const CONTROL_RED_GAIN: i32 = 8;
pub const CONTROL_BLUE_GAIN: i32 = 9;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn control_number(value: &ControlValue) -> f64 {
    match value {
        ControlValue::Bool(b) => f64::from(u8::from(*b)),
        ControlValue::Int(v) => f64::from(*v),
        ControlValue::Uint(v) => f64::from(*v),
        ControlValue::Float(v) => f64::from(*v),
        _ => f64::NAN,
    }
}

struct Latest {
    seq: u64,
    taken: u64,
    jpeg: Vec<u8>,
}

/// A frame for the processed stream: a share of the frame (released as soon as its luma is
/// copied) and what to draw on it.
struct Job {
    frame: FrameLease,
    overlays: Vec<Overlay>,
    draw: bool,
    divisor: u32,
}

/// The processed stream's encoder thread and its hand-offs.
struct Processed {
    job: Mutex<Option<Job>>,
    job_ready: Condvar,
    out: Mutex<Latest>,
    out_ready: Condvar,
    /// CLOCK_MONOTONIC ns until which someone watches (see [`WATCH_TIMEOUT_NS`]).
    watched_until: AtomicU64,
    /// CLOCK_MONOTONIC ns of the last frame handed to the encoder (the rate cap).
    last_offer: AtomicU64,
    stop: AtomicBool,
}

impl Processed {
    fn new() -> Self {
        Self {
            job: Mutex::new(None),
            job_ready: Condvar::new(),
            out: Mutex::new(Latest { seq: 0, taken: 0, jpeg: Vec::new() }),
            out_ready: Condvar::new(),
            watched_until: AtomicU64::new(0),
            last_offer: AtomicU64::new(0),
            stop: AtomicBool::new(false),
        }
    }

    /// Whether the vision thread should hand this frame over (watched, and due by the rate cap).
    fn due(&self, now: u64) -> bool {
        if now > self.watched_until.load(Ordering::Relaxed) {
            return false;
        }
        let interval = (1e9 / f64::from(STREAM_FPS)) as u64;
        let last = self.last_offer.load(Ordering::Relaxed);
        // A little slack, so a 60 fps camera is not halved to 20 fps by jitter.
        if now.saturating_sub(last) + interval / 8 < interval {
            return false;
        }
        self.last_offer.store(now, Ordering::Relaxed);
        true
    }

    fn put(&self, job: Job) {
        let replaced = lock(&self.job).replace(job);
        self.job_ready.notify_one();
        drop(replaced);
    }

    fn run(&self) {
        let mut encoder = match PlanarJpegEncoder::new(JpegBackend::Auto, PROCESSED_QUALITY) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("photonstyx: no JPEG encoder for the processed stream: {e}");
                return;
            }
        };
        // Lower than the vision thread (as Styx's preview thread is).
        // SAFETY: setpriority on this thread (who = 0 is the calling thread on Linux).
        unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 10) };
        let (mut luma, mut u, mut v, mut jpeg) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        loop {
            let job = {
                let mut slot = lock(&self.job);
                loop {
                    if self.stop.load(Ordering::Acquire) {
                        return;
                    }
                    if let Some(job) = slot.take() {
                        break job;
                    }
                    slot = self.job_ready.wait(slot).unwrap_or_else(|e| e.into_inner());
                }
            };
            let Some((width, height, scale)) = copy_stream_luma(&job.frame, job.divisor, &mut luma) else {
                continue;
            };
            // The frame's buffer goes back to the camera before the drawing and encoding.
            drop(job.frame);
            let mut canvas = Canvas::grey(std::mem::take(&mut luma), width, height, std::mem::take(&mut u), std::mem::take(&mut v));
            if job.draw {
                overlay::draw(&mut canvas, &job.overlays, scale);
            }
            let (cw, _) = Canvas::chroma_size(width, height);
            let input = JpegInput::I420 {
                y: &canvas.y,
                u: &canvas.u,
                v: &canvas.v,
                width,
                height,
                y_stride: width,
                c_stride: cw,
            };
            let encoded = encoder.encode(&input, &mut jpeg);
            (luma, u, v) = (canvas.y, canvas.u, canvas.v);
            if let Err(e) = encoded {
                eprintln!("photonstyx: processed stream encode failed: {e}");
                continue;
            }
            let mut out = lock(&self.out);
            out.seq += 1;
            std::mem::swap(&mut out.jpeg, &mut jpeg);
            drop(out);
            self.out_ready.notify_all();
        }
    }
}

/// A board detection for the record: found or not. A failed detection is logged and reported as
/// not found, so one bad frame never stops the stream.
fn detect_board(board: &Board, frame: &FrameLease) -> BoardRecord {
    match calibrate::detect(board, frame) {
        Ok(Some(found)) => BoardRecord {
            found: true,
            ids: found.ids.iter().map(|&id| id as i32).collect(),
            corners: found.corners.iter().map(|c| [c.u, c.v]).collect(),
        },
        Ok(None) => BoardRecord::default(),
        Err(e) => {
            eprintln!("photonstyx: board detection failed: {e}");
            BoardRecord::default()
        }
    }
}

/// The luma of the pyramid level the stream divisor asks for (at most [`PROCESSED_MAX_WIDTH`]
/// wide; a further halving on the CPU when no level is small enough), packed into `out`.
/// Returns the size and the scale from full-frame pixels.
fn copy_stream_luma(frame: &FrameLease, divisor: u32, out: &mut Vec<u8>) -> Option<(usize, usize, f64)> {
    let full_width = frame.meta().format.resolution.width.get() as usize;
    let mut want = divisor.max(1) as usize;
    while full_width / want > PROCESSED_MAX_WIDTH {
        want *= 2;
    }
    // The largest level no bigger than wanted (level 1 is the ISP's half).
    let mut level = 0u8;
    let mut source = frame;
    while (1usize << (level + 1)) <= want {
        match frame.pyramid_level(level + 1) {
            Some(next) => {
                level += 1;
                source = next;
            }
            None => break,
        }
    }
    let rows = source.luma_rows().ok()?;
    let step = (want >> level).max(1);
    let (sw, sh) = (rows.row_bytes(), rows.len());
    let (w, h) = (sw / step, sh / step);
    if w == 0 || h == 0 {
        return None;
    }
    out.clear();
    out.reserve(w * h);
    for (y, row) in rows.iter().enumerate() {
        if y % step != 0 || y / step >= h {
            continue;
        }
        let data = &row.data()[..sw];
        if step == 1 {
            out.extend_from_slice(&data[..w]);
        } else {
            out.extend(data.iter().step_by(step).take(w));
        }
    }
    let scale = w as f64 / full_width as f64;
    Some((w, h, scale))
}

struct Capture {
    frames: Option<Frames>,
    detector: Option<Detector>,
    processed_enabled: bool,
    draw: bool,
    divisor: u32,
    sequence: i64,
    /// A board detection asked for on the next frame (taken by `process`).
    board_request: Option<Board>,
}

pub struct Session {
    capture: Mutex<Capture>,
    /// Where the opened mode's frames sit in the sensor's full field (`x, y, width, height`),
    /// when the mode is a window of it (e.g. 1280x480 inside 1280x800); read from the first frame.
    crop: Option<[i32; 4]>,
    controls: StandardControls,
    gains: Mutex<(f32, f32)>,
    preview: Mutex<Option<Preview>>,
    /// The raw stream's subscriber, while someone watches it (the preview encodes only then).
    raw: Mutex<Option<PreviewSubscriber>>,
    raw_watched_until: AtomicU64,
    processed: Arc<Processed>,
    processed_thread: Mutex<Option<JoinHandle<()>>>,
}

fn clock_ns(clock: libc::clockid_t) -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: clock_gettime writes one timespec through a valid pointer.
    unsafe { libc::clock_gettime(clock, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

pub fn monotonic_ns() -> u64 {
    clock_ns(libc::CLOCK_MONOTONIC)
}

pub fn boottime_ns() -> u64 {
    clock_ns(libc::CLOCK_BOOTTIME)
}

/// Names of the cameras Styx sees, as `open` takes them.
pub fn list_cameras() -> Vec<String> {
    let probe = styx::probe_all_with_errors_with_config(
        &StyxConfig::default().timestamp_clock(ClockSource::Boottime),
    );
    probe.devices.iter().map(|d| d.identity.display.clone()).collect()
}

impl Session {
    /// Open `camera` (a name from [`list_cameras`]; empty takes the first) at exactly `width` x
    /// `height`, at `fps` (0: the sensor's natural rate), as grey frames with the ISP's half-size
    /// pyramid level. Detection reads level 1; refinement reads the full frame.
    pub fn open(camera: &str, width: u32, height: u32, fps: u32) -> Result<Self, String> {
        let probe = styx::probe_all_with_errors_with_config(
            &StyxConfig::default().timestamp_clock(ClockSource::Boottime),
        );
        let device = probe
            .devices
            .iter()
            .find(|d| camera.is_empty() || d.identity.display == camera)
            .ok_or_else(|| format!("no camera named '{camera}' (found {} devices)", probe.devices.len()))?;
        // The newest frame only (Styx's default delivery): a frame the vision thread has not
        // taken when the next one arrives is dropped, never queued.
        let mut request = Frames::gray().size(width, height).pyramid(1).latest();
        if fps > 0 {
            request = request.fps(fps);
        }
        let plan = request
            .plan_best(std::slice::from_ref(device))
            .map_err(|e| format!("cannot plan '{}': {e}", device.identity.display))?;
        let mut frames = plan
            .start()
            .map_err(|e| format!("cannot start '{}': {e}", device.identity.display))?;
        // A windowed mode reports its place in the full field on every frame; take it from the
        // first one so the caller can place a full-field calibration before detection starts.
        let crop = match frames.next_frame(Duration::from_millis(2000)) {
            RecvOutcome::Data(lease) => lease
                .meta()
                .crop
                .map(|c| [c.x as i32, c.y as i32, c.width as i32, c.height as i32]),
            _ => None,
        };
        let controls = frames.standard_controls();
        let preview = Preview::new(
            PreviewConfig::new()
                .name(format!("photonstyx-{}", device.identity.display))
                .size(PREVIEW_MAX_SIZE.0, PREVIEW_MAX_SIZE.1)
                .max_fps(STREAM_FPS)
                .quality(PREVIEW_QUALITY),
        )
        .map_err(|e| format!("cannot start the preview: {e}"))?;
        let processed = Arc::new(Processed::new());
        let thread = {
            let processed = processed.clone();
            std::thread::Builder::new()
                .name("photonstyx-processed".into())
                .spawn(move || processed.run())
                .map_err(|e| format!("cannot start the processed stream: {e}"))?
        };
        Ok(Self {
            crop,
            capture: Mutex::new(Capture {
                frames: Some(frames),
                detector: None,
                processed_enabled: false,
                draw: true,
                divisor: 1,
                sequence: 0,
                board_request: None,
            }),
            controls,
            gains: Mutex::new((1.0, 1.0)),
            preview: Mutex::new(Some(preview)),
            raw: Mutex::new(None),
            raw_watched_until: AtomicU64::new(0),
            processed,
            processed_thread: Mutex::new(Some(thread)),
        })
    }

    /// The opened mode's window in the sensor's full field (`x, y, width, height`), or `None` for
    /// a full-field mode.
    pub fn crop(&self) -> Option<[i32; 4]> {
        self.crop
    }

    /// Stop the capture, then the streams (their subscribers end).
    pub fn stop(&self) {
        if let Some(frames) = lock(&self.capture).frames.take() {
            frames.stop();
        }
        lock(&self.raw).take();
        if let Some(preview) = lock(&self.preview).take() {
            preview.stop();
        }
        self.processed.stop.store(true, Ordering::Release);
        self.processed.job_ready.notify_all();
        lock(&self.processed.job).take();
        if let Some(thread) = lock(&self.processed_thread).take() {
            let _ = thread.join();
        }
    }

    /// Replace the detector and the stream settings.
    pub fn configure(&self, config: &PipelineConfig) -> Result<(), String> {
        let detector = Detector::new(config)?;
        let mut capture = lock(&self.capture);
        capture.detector = Some(detector);
        capture.processed_enabled = config.processed_stream;
        capture.draw = config.draw;
        capture.divisor = config.stream_divisor.max(1);
        Ok(())
    }

    /// Detect `board` on the next frame (calibration snapshots). The result rides in that frame's
    /// record (see `pack.rs`); a later request replaces an unanswered one.
    pub fn request_board(&self, board: Board) {
        lock(&self.capture).board_request = Some(board);
    }

    /// Apply a standard control (`CONTROL_*`); returns the value in effect.
    pub fn set_control(&self, kind: i32, value: f64) -> Result<f64, String> {
        let applied = match kind {
            CONTROL_EXPOSURE_US => self.controls.set_exposure_us(value.max(0.0) as u32),
            CONTROL_GAIN => self.controls.set_gain(value as f32),
            CONTROL_AE => self.controls.set_ae(value != 0.0),
            CONTROL_EV => self.controls.set_ev(value as f32),
            CONTROL_FPS => self.controls.set_fps(value as f32),
            CONTROL_AWB => self.controls.set_awb(value != 0.0),
            CONTROL_COLOUR_TEMPERATURE => self.controls.set_colour_temperature(value.max(0.0) as u32),
            CONTROL_RED_GAIN | CONTROL_BLUE_GAIN => {
                // The gains are set as a pair; the other one keeps its last value.
                let (red, blue) = {
                    let mut gains = lock(&self.gains);
                    if kind == CONTROL_RED_GAIN {
                        gains.0 = value as f32;
                    } else {
                        gains.1 = value as f32;
                    }
                    *gains
                };
                self.controls.set_colour_gains(red, blue).map(|(r, _)| r)
            }
            other => return Err(format!("unknown control {other}")),
        }
        .map_err(|e| format!("control {kind}: {e}"))?;
        Ok(control_number(&applied.value))
    }

    /// Wait up to `timeout` for the next frame and detect on it. `Ok(None)` on timeout.
    pub fn process(&self, timeout: Duration) -> Result<Option<Vec<u8>>, String> {
        let mut capture = lock(&self.capture);
        let frames = capture.frames.as_mut().ok_or("the camera is stopped")?;
        let lease = match frames.next_frame(timeout) {
            RecvOutcome::Data(lease) => lease,
            RecvOutcome::Empty => return Ok(None),
            RecvOutcome::Closed => return Err("the camera stopped".into()),
        };
        let taken = monotonic_ns();
        // Hops are CLOCK_MONOTONIC; the result is on CLOCK_BOOTTIME (PhotonVision's latency).
        let boot_offset = boottime_ns() as i64 - monotonic_ns() as i64;
        let boot = |mono: Option<u64>| mono.map_or(0, |ns| ns as i64 + boot_offset);

        let meta = lease.meta();
        let (width, height) = (
            meta.format.resolution.width.get() as i32,
            meta.format.resolution.height.get() as i32,
        );
        // When the scene was imaged: the middle of the frame's exposure (the sensor timestamp is the
        // start of readout, after the exposure). Robot code aligns vision with odometry on it.
        let capture_nanos = meta
            .exposure_midpoint_in(TimestampClock::Boottime)
            .or_else(|| meta.timestamp_in(TimestampClock::Boottime))
            .unwrap_or(0) as i64;
        let hops = meta.hops;
        capture.sequence += 1;
        let sequence = capture.sequence;

        let detect_start = monotonic_ns();
        let detections = match capture.detector.as_mut() {
            Some(detector) => detector.process(&lease)?,
            None => Detections::default(),
        };
        let board = capture.board_request.take().map(|board| detect_board(&board, &lease));
        let detect_end = monotonic_ns();

        // The streams: hand out shares of the frame (no copies here) and let go of it. The PiSP
        // has six output buffers and a held frame keeps one (see pv_capture's buffer rule); the
        // stream threads release theirs as soon as they have scaled or copied it.
        let lease = lease.into_shareable();
        if capture.processed_enabled
            && self.processed.due(detect_end)
            && let Some(share) = lease.share()
        {
            self.processed.put(Job {
                frame: share,
                overlays: detections.overlays.clone(),
                draw: capture.draw,
                divisor: capture.divisor,
            });
        }
        if detect_end > self.raw_watched_until.load(Ordering::Relaxed) {
            // Nobody watches the raw stream: drop its subscriber (when the stream thread is not
            // waiting on it), so the preview stops encoding.
            if let Ok(mut raw) = self.raw.try_lock() {
                raw.take();
            }
        }
        if let Some(preview) = lock(&self.preview).as_ref() {
            preview.offer_owned(lease);
        }

        let mut stages = [0i64; STAGES];
        stages[0] = boot(hops.get(Hop::Sensor));
        stages[1] = boot(hops.get(Hop::Dequeued));
        stages[2] = boot(hops.get(Hop::IspDone));
        stages[3] = boot(hops.get(Hop::Queued));
        stages[4] = boot(Some(taken));
        stages[5] = boot(Some(detect_start));
        stages[6] = boot(Some(detect_end));
        let mut record = FrameRecord {
            sequence,
            capture_nanos,
            processing_nanos: (detect_end - detect_start) as i64,
            stages,
            width,
            height,
            multi: detections.multi,
            targets: detections.targets,
            board,
        };
        record.stages[7] = boot(Some(monotonic_ns()));
        Ok(Some(pack::encode(&record)))
    }

    /// The next raw JPEG (grey, from the preview), waiting up to `timeout`.
    pub fn raw_jpeg(&self, timeout: Duration) -> Option<Vec<u8>> {
        self.raw_watched_until
            .store(monotonic_ns() + WATCH_TIMEOUT_NS, Ordering::Relaxed);
        let mut raw = lock(&self.raw);
        if raw.is_none() {
            *raw = lock(&self.preview).as_ref().map(Preview::subscribe);
        }
        raw.as_mut()?.recv(timeout).map(|frame| frame.jpeg.to_vec())
    }

    /// The next processed JPEG (the frame with the detections drawn), waiting up to `timeout`.
    pub fn processed_jpeg(&self, timeout: Duration) -> Option<Vec<u8>> {
        self.processed
            .watched_until
            .store(monotonic_ns() + WATCH_TIMEOUT_NS, Ordering::Relaxed);
        let guard = lock(&self.processed.out);
        let (mut latest, _) = self
            .processed
            .out_ready
            .wait_timeout_while(guard, timeout, |l| l.seq == l.taken)
            .unwrap_or_else(|e| e.into_inner());
        if latest.seq == latest.taken {
            return None;
        }
        latest.taken = latest.seq;
        Some(latest.jpeg.clone())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.stop();
    }
}
