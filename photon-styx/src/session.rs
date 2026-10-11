//! One camera: a Styx grey capture (the kernel-driven sensor's 3A-processed luma, through the
//! PiSP ISP), the Eidos detector, the raw JPEG preview and the processed JPEG. Follows the
//! reference flow in Styx's `pv_capture` (styx-cm5-prep): frames are offered to the preview
//! with `offer_owned`, so a lease is released as soon as the detector has read it.

use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use styx::capture_api::StandardControls;
use styx::prelude::*;
use styx::preview::{Preview, PreviewConfig, PreviewSubscriber};

use crate::config::PipelineConfig;
use crate::detect::{Detections, Detector};
use crate::overlay;
use crate::pack::{self, FrameRecord, MultiRecord, TargetRecord};

/// The raw preview: at most 640x400, 30 fps, JPEG quality 70 (encoded off the vision thread).
const PREVIEW_MAX_SIZE: (u32, u32) = (640, 400);
const PREVIEW_FPS: f32 = 30.0;
const PREVIEW_QUALITY: u8 = 70;
const PROCESSED_QUALITY: u8 = 70;

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

#[derive(Default)]
struct Latest {
    seq: u64,
    taken: u64,
    jpeg: Vec<u8>,
}

struct Capture {
    frames: Option<Frames>,
    detector: Option<Detector>,
    processed_enabled: bool,
    sequence: i64,
}

pub struct Session {
    capture: Mutex<Capture>,
    controls: StandardControls,
    gains: Mutex<(f32, f32)>,
    preview: Mutex<Option<Preview>>,
    raw: Mutex<PreviewSubscriber>,
    processed: Mutex<Latest>,
    processed_ready: Condvar,
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
    /// `height`, at `fps` (0: the sensor's natural rate), as grey frames with a half-size pyramid level (level 1: the ISP's
    /// half, level 2: a CPU quarter). Detection reads level 1; refinement reads the full frame.
    pub fn open(camera: &str, width: u32, height: u32, fps: u32) -> Result<Self, String> {
        let probe = styx::probe_all_with_errors_with_config(
            &StyxConfig::default().timestamp_clock(ClockSource::Boottime),
        );
        let device = probe
            .devices
            .iter()
            .find(|d| camera.is_empty() || d.identity.display == camera)
            .ok_or_else(|| format!("no camera named '{camera}' (found {} devices)", probe.devices.len()))?;
        // The size is exact. The rate is the sensor's natural one unless `fps` is non-zero: exact
        // rates are not reliable on the kernel-driven sensor yet (30 asked, 21 delivered), so
        // PhotonVision asks for 0.
        let mut request = Frames::gray().size(width, height).pyramid(2);
        if fps > 0 {
            request = request.fps(fps);
        }
        let plan = request
            .plan_best(std::slice::from_ref(device))
            .map_err(|e| format!("cannot plan '{}': {e}", device.identity.display))?;
        let frames = plan
            .start()
            .map_err(|e| format!("cannot start '{}': {e}", device.identity.display))?;
        let controls = frames.standard_controls();
        let preview = Preview::new(
            PreviewConfig::new()
                .name(format!("photonstyx-{}", device.identity.display))
                .size(PREVIEW_MAX_SIZE.0, PREVIEW_MAX_SIZE.1)
                .max_fps(PREVIEW_FPS)
                .quality(PREVIEW_QUALITY),
        )
        .map_err(|e| format!("cannot start the preview: {e}"))?;
        let raw = preview.subscribe();
        Ok(Self {
            capture: Mutex::new(Capture {
                frames: Some(frames),
                detector: None,
                processed_enabled: false,
                sequence: 0,
            }),
            controls,
            gains: Mutex::new((1.0, 1.0)),
            preview: Mutex::new(Some(preview)),
            raw: Mutex::new(raw),
            processed: Mutex::new(Latest::default()),
            processed_ready: Condvar::new(),
        })
    }

    /// Stop the capture, then the preview (its subscribers end).
    pub fn stop(&self) {
        if let Some(frames) = lock(&self.capture).frames.take() {
            frames.stop();
        }
        if let Some(preview) = lock(&self.preview).take() {
            preview.stop();
        }
    }

    /// Replace the detector and the stream settings.
    pub fn configure(&self, config: &PipelineConfig) -> Result<(), String> {
        let detector = Detector::new(config)?;
        let mut capture = lock(&self.capture);
        capture.detector = Some(detector);
        capture.processed_enabled = config.processed_stream;
        Ok(())
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
        let started = Instant::now();

        let meta = lease.meta();
        let (width, height) = (
            meta.format.resolution.width.get() as i32,
            meta.format.resolution.height.get() as i32,
        );
        let capture_nanos = meta.timestamp_in(TimestampClock::Boottime).unwrap_or(0) as i64;
        capture.sequence += 1;
        let sequence = capture.sequence;

        let detections = match capture.detector.as_mut() {
            Some(detector) => detector.process(&lease)?,
            None => Detections {
                targets: Vec::<TargetRecord>::new(),
                multi: MultiRecord::default(),
                quads: Vec::new(),
            },
        };
        if capture.processed_enabled {
            self.publish_processed(&lease, &detections.quads)?;
        }
        // Hand the lease to the preview and let go of it here: the PiSP has six output buffers
        // and a held frame keeps one of them (see pv_capture's buffer rule).
        if let Some(preview) = lock(&self.preview).as_ref() {
            preview.offer_owned(lease.into_shareable());
        }

        let record = FrameRecord {
            sequence,
            capture_nanos,
            processing_nanos: started.elapsed().as_nanos() as i64,
            width,
            height,
            multi: detections.multi,
            targets: detections.targets,
        };
        Ok(Some(pack::encode(&record)))
    }

    /// The next raw JPEG (grey, from the preview), waiting up to `timeout`.
    pub fn raw_jpeg(&self, timeout: Duration) -> Option<Vec<u8>> {
        lock(&self.raw).recv(timeout).map(|frame| frame.jpeg.to_vec())
    }

    /// The next processed JPEG (luma with the tag outlines), waiting up to `timeout`.
    pub fn processed_jpeg(&self, timeout: Duration) -> Option<Vec<u8>> {
        let guard = lock(&self.processed);
        let (mut latest, _) = self
            .processed_ready
            .wait_timeout_while(guard, timeout, |l| l.seq == l.taken)
            .unwrap_or_else(|e| e.into_inner());
        if latest.seq == latest.taken {
            return None;
        }
        latest.taken = latest.seq;
        Some(latest.jpeg.clone())
    }

    fn publish_processed(&self, lease: &FrameLease, quads: &[[f64; 8]]) -> Result<(), String> {
        let meta = lease.meta();
        let width = meta.format.resolution.width.get() as usize;
        let height = meta.format.resolution.height.get() as usize;
        let rows = lease.luma_rows().map_err(|e| format!("no luma plane: {e:?}"))?;
        let mut luma = Vec::with_capacity(width * height);
        for row in rows.iter() {
            luma.extend_from_slice(&row.data()[..width]);
        }
        overlay::draw_quads(&mut luma, width, height, width, quads, 255);
        let jpeg = overlay::encode_grey_jpeg(&luma, width, height, width, PROCESSED_QUALITY)?;
        let mut latest = lock(&self.processed);
        latest.seq += 1;
        latest.jpeg = jpeg;
        drop(latest);
        self.processed_ready.notify_all();
        Ok(())
    }
}
