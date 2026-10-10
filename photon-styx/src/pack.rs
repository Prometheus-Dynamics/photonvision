//! The per-frame result as little-endian bytes, decoded by `org.photonvision.styx.StyxResult`.
//!
//! Layout (version 1), in order:
//! - header: `i32 version`, `i64 sequence`, `i64 captureNanos` (CLOCK_BOOTTIME, 0 if unknown),
//!   `i64 processingNanos`, `i32 width`, `i32 height`, `i32 targetCount`;
//! - multi-tag: `i32 valid`, `i32 inlierTags`, `f64 rmsPx`, `f64 ambiguity`, `f64[7] pose`
//!   (WPILib camera pose in the field frame: x, y, z, qw, qx, qy, qz);
//! - `targetCount` records of `i32 id`, `i32 hamming`, `i32 usedInMulti`, `i32 hasAlt`,
//!   `f64[8] corners` (refined, image pixels, marker order), `f64[7] bestPose`,
//!   `f64[7] altPose` (both optical camera-from-tag: x, y, z, qw, qx, qy, qz),
//!   `f64 ambiguity`, `f64 rmsBestPx`, `f64 rmsAltPx`.

pub const VERSION: i32 = 1;
pub const MAX_TARGETS: usize = 64;
pub const HEADER_BYTES: usize = 4 + 8 + 8 + 8 + 4 + 4 + 4;
pub const MULTI_BYTES: usize = 4 + 4 + 8 + 8 + 7 * 8;
pub const TARGET_BYTES: usize = 4 * 4 + 8 * 8 + 7 * 8 + 7 * 8 + 3 * 8;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MultiRecord {
    pub valid: bool,
    pub inlier_tags: i32,
    pub rms_px: f64,
    pub ambiguity: f64,
    pub pose: [f64; 7],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetRecord {
    pub id: i32,
    pub hamming: i32,
    pub used_in_multi: bool,
    pub corners: [f64; 8],
    pub best: [f64; 7],
    pub alt: Option<[f64; 7]>,
    pub ambiguity: f64,
    pub rms_best_px: f64,
    pub rms_alt_px: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameRecord {
    pub sequence: i64,
    pub capture_nanos: i64,
    pub processing_nanos: i64,
    pub width: i32,
    pub height: i32,
    pub multi: MultiRecord,
    pub targets: Vec<TargetRecord>,
}

struct Writer(Vec<u8>);

impl Writer {
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64s(&mut self, v: &[f64]) {
        for &x in v {
            self.f64(x);
        }
    }
}

/// Encode a frame record. Targets past [`MAX_TARGETS`] are dropped.
pub fn encode(record: &FrameRecord) -> Vec<u8> {
    let targets = &record.targets[..record.targets.len().min(MAX_TARGETS)];
    let mut w = Writer(Vec::with_capacity(HEADER_BYTES + MULTI_BYTES + targets.len() * TARGET_BYTES));
    w.i32(VERSION);
    w.i64(record.sequence);
    w.i64(record.capture_nanos);
    w.i64(record.processing_nanos);
    w.i32(record.width);
    w.i32(record.height);
    w.i32(targets.len() as i32);
    w.i32(record.multi.valid as i32);
    w.i32(record.multi.inlier_tags);
    w.f64(record.multi.rms_px);
    w.f64(record.multi.ambiguity);
    w.f64s(&record.multi.pose);
    for t in targets {
        w.i32(t.id);
        w.i32(t.hamming);
        w.i32(t.used_in_multi as i32);
        w.i32(t.alt.is_some() as i32);
        w.f64s(&t.corners);
        w.f64s(&t.best);
        w.f64s(&t.alt.unwrap_or([0.0; 7]));
        w.f64(t.ambiguity);
        w.f64(t.rms_best_px);
        w.f64(t.rms_alt_px);
    }
    w.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i32_at(b: &[u8], at: usize) -> i32 {
        i32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }
    fn f64_at(b: &[u8], at: usize) -> f64 {
        f64::from_le_bytes(b[at..at + 8].try_into().unwrap())
    }

    #[test]
    fn packs_header_multi_and_targets_at_fixed_offsets() {
        let record = FrameRecord {
            sequence: 7,
            capture_nanos: 123_456,
            processing_nanos: 900,
            width: 1280,
            height: 800,
            multi: MultiRecord {
                valid: true,
                inlier_tags: 2,
                rms_px: 0.5,
                ambiguity: 0.25,
                pose: [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            },
            targets: vec![
                TargetRecord {
                    id: 11,
                    hamming: 1,
                    used_in_multi: true,
                    corners: [1.0; 8],
                    best: [0.1; 7],
                    alt: Some([0.2; 7]),
                    ambiguity: 0.3,
                    rms_best_px: 0.4,
                    rms_alt_px: 0.6,
                },
                TargetRecord {
                    id: 12,
                    hamming: 0,
                    used_in_multi: false,
                    corners: [2.0; 8],
                    best: [0.5; 7],
                    alt: None,
                    ambiguity: 0.0,
                    rms_best_px: 0.0,
                    rms_alt_px: 0.0,
                },
            ],
        };
        let bytes = encode(&record);
        assert_eq!(bytes.len(), HEADER_BYTES + MULTI_BYTES + 2 * TARGET_BYTES);
        assert_eq!(i32_at(&bytes, 0), VERSION);
        assert_eq!(i32_at(&bytes, 36), 2); // target count
        let multi = HEADER_BYTES;
        assert_eq!(i32_at(&bytes, multi), 1);
        assert_eq!(f64_at(&bytes, multi + 16), 0.25);
        let t0 = HEADER_BYTES + MULTI_BYTES;
        assert_eq!(i32_at(&bytes, t0), 11);
        assert_eq!(i32_at(&bytes, t0 + 12), 1); // has alt
        assert_eq!(f64_at(&bytes, t0 + 16), 1.0); // first corner
        let alt = t0 + 16 + 64 + 56;
        assert_eq!(f64_at(&bytes, alt), 0.2);
        let t1 = t0 + TARGET_BYTES;
        assert_eq!(i32_at(&bytes, t1), 12);
        assert_eq!(i32_at(&bytes, t1 + 12), 0);
    }

    #[test]
    fn caps_the_target_count() {
        let mut record = FrameRecord {
            sequence: 0,
            capture_nanos: 0,
            processing_nanos: 0,
            width: 0,
            height: 0,
            multi: MultiRecord::default(),
            targets: vec![],
        };
        let one = TargetRecord {
            id: 0,
            hamming: 0,
            used_in_multi: false,
            corners: [0.0; 8],
            best: [0.0; 7],
            alt: None,
            ambiguity: 0.0,
            rms_best_px: 0.0,
            rms_alt_px: 0.0,
        };
        record.targets = vec![one; MAX_TARGETS + 5];
        let bytes = encode(&record);
        assert_eq!(bytes.len(), HEADER_BYTES + MULTI_BYTES + MAX_TARGETS * TARGET_BYTES);
    }
}
