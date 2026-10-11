//! The processed stream: the frame's luma as a colour (YUV 4:2:0) image with each detected tag
//! drawn on it, as PhotonVision's AprilTag pipeline draws its output: the tag outline, its centre
//! and id, and in 3D mode the box standing on the tag. Encoded off the vision thread.

use crate::detect::Overlay;

/// A colour as JFIF (full-range BT.601) YUV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Yuv(pub u8, pub u8, pub u8);

impl Yuv {
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        let (r, g, b) = (f64::from(r), f64::from(g), f64::from(b));
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let u = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
        let v = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        let c = |x: f64| x.round().clamp(0.0, 255.0) as u8;
        Self(c(y), c(u), c(v))
    }
}

/// PhotonVision's overlay colours: the outline green, the 3D box's pillars blue and its far face
/// orange, the id red, the centre yellow.
pub fn outline() -> Yuv {
    Yuv::rgb(0, 255, 0)
}
pub fn pillar() -> Yuv {
    Yuv::rgb(0, 128, 255)
}
pub fn top_face() -> Yuv {
    Yuv::rgb(255, 140, 0)
}
pub fn label() -> Yuv {
    Yuv::rgb(255, 40, 40)
}
pub fn centre() -> Yuv {
    Yuv::rgb(255, 255, 0)
}

/// A YUV 4:2:0 image, planes packed (no row padding).
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl Canvas {
    pub fn chroma_size(width: usize, height: usize) -> (usize, usize) {
        (width.div_ceil(2), height.div_ceil(2))
    }

    /// A grey canvas over `luma` (`width` x `height`, packed), reusing `u` and `v`.
    pub fn grey(luma: Vec<u8>, width: usize, height: usize, mut u: Vec<u8>, mut v: Vec<u8>) -> Self {
        let (cw, ch) = Self::chroma_size(width, height);
        u.clear();
        u.resize(cw * ch, 128);
        v.clear();
        v.resize(cw * ch, 128);
        Self { width, height, y: luma, u, v }
    }

    #[inline]
    fn put(&mut self, x: i64, y: i64, c: Yuv) {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return;
        }
        let (x, y) = (x as usize, y as usize);
        self.y[y * self.width + x] = c.0;
        let cw = self.width.div_ceil(2);
        let i = (y / 2) * cw + x / 2;
        self.u[i] = c.1;
        self.v[i] = c.2;
    }

    /// A `size` x `size` square of colour `c` at (`x`, `y`) (top-left).
    fn dot(&mut self, x: i64, y: i64, size: i64, c: Yuv) {
        for dy in 0..size {
            for dx in 0..size {
                self.put(x + dx, y + dy, c);
            }
        }
    }

    /// A line `thickness` pixels wide.
    pub fn line(&mut self, a: (f64, f64), b: (f64, f64), thickness: i64, c: Yuv) {
        if !(a.0.is_finite() && a.1.is_finite() && b.0.is_finite() && b.1.is_finite()) {
            return;
        }
        // Lines far off the image (a bad projection) are not drawn.
        let limit = 4.0 * (self.width + self.height) as f64;
        if [a.0, a.1, b.0, b.1].iter().any(|v| v.abs() > limit) {
            return;
        }
        let steps = (b.0 - a.0).abs().max((b.1 - a.1).abs()).ceil().max(1.0) as usize;
        let half = thickness / 2;
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let x = (a.0 + (b.0 - a.0) * t).round() as i64;
            let y = (a.1 + (b.1 - a.1) * t).round() as i64;
            self.dot(x - half, y - half, thickness, c);
        }
    }

    /// The closed polygon through `pts` (`[u0, v0, u1, v1, ...]`).
    pub fn polygon(&mut self, pts: &[f64], thickness: i64, c: Yuv) {
        let n = pts.len() / 2;
        for i in 0..n {
            let j = (i + 1) % n;
            self.line((pts[2 * i], pts[2 * i + 1]), (pts[2 * j], pts[2 * j + 1]), thickness, c);
        }
    }

    /// `text` (digits, `-` and a few letters) at (`x`, `y`) (top-left), each font pixel `scale`
    /// pixels, on a dark backing so it reads on any background.
    pub fn text(&mut self, x: i64, y: i64, text: &str, scale: i64, c: Yuv) {
        let advance = 4 * scale;
        let w = text.chars().count() as i64 * advance + scale;
        for dy in 0..(5 * scale + 2 * scale) {
            for dx in 0..w {
                self.put(x - scale + dx, y - scale + dy, Yuv(0, 128, 128));
            }
        }
        for (k, ch) in text.chars().enumerate() {
            let glyph = glyph(ch);
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..3 {
                    if bits & (0b100 >> col) != 0 {
                        self.dot(x + k as i64 * advance + col * scale, y + row as i64 * scale, scale, c);
                    }
                }
            }
        }
    }
}

/// 3x5 glyphs, one row per entry, bit 2 the left column.
fn glyph(ch: char) -> [u8; 5] {
    match ch {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        _ => [0; 5],
    }
}

/// Draw `overlays` (full-frame pixels) on `canvas`, which is the frame scaled by `scale`.
pub fn draw(canvas: &mut Canvas, overlays: &[Overlay], scale: f64) {
    let thick = if canvas.width >= 960 { 3 } else { 2 };
    let text_scale = if canvas.width >= 960 { 4 } else { 3 };
    let s = |pts: &[f64; 8]| pts.map(|v| v * scale);
    for o in overlays {
        let base = s(&o.corners);
        if let Some(top) = o.top.as_ref().map(s) {
            for i in 0..4 {
                canvas.line((base[2 * i], base[2 * i + 1]), (top[2 * i], top[2 * i + 1]), thick, pillar());
            }
            canvas.polygon(&top, thick, top_face());
        }
        canvas.polygon(&base, thick, outline());
        let cx = (base[0] + base[2] + base[4] + base[6]) / 4.0;
        let cy = (base[1] + base[3] + base[5] + base[7]) / 4.0;
        if cx.is_finite() && cy.is_finite() {
            canvas.line((cx - 4.0, cy), (cx + 4.0, cy), thick.min(2), centre());
            canvas.line((cx, cy - 4.0), (cx, cy + 4.0), thick.min(2), centre());
            // The id above the tag's highest corner.
            let top_y = [base[1], base[3], base[5], base[7]].into_iter().fold(f64::INFINITY, f64::min);
            let id = o.id.to_string();
            let w = id.len() as f64 * 4.0 * text_scale as f64;
            canvas.text(
                (cx - w / 2.0).round() as i64,
                (top_y - 8.0 * text_scale as f64).round() as i64,
                &id,
                text_scale,
                label(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colours_convert_like_jfif() {
        assert_eq!(Yuv::rgb(0, 0, 0), Yuv(0, 128, 128));
        assert_eq!(Yuv::rgb(255, 255, 255), Yuv(255, 128, 128));
        let g = Yuv::rgb(0, 255, 0);
        assert!(g.1 < 128 && g.2 < 128, "green has low U and V: {g:?}");
    }

    #[test]
    fn draws_a_coloured_outline_box_and_id() {
        let (w, h) = (64usize, 48usize);
        let mut c = Canvas::grey(vec![0u8; w * h], w, h, Vec::new(), Vec::new());
        let quad = [10.0, 20.0, 40.0, 20.0, 40.0, 40.0, 10.0, 40.0];
        let top = [14.0, 14.0, 44.0, 14.0, 44.0, 34.0, 14.0, 34.0];
        draw(&mut c, &[Overlay { id: 7, corners: quad, top: Some(top) }], 1.0);
        let at = |x: usize, y: usize| (c.y[y * w + x], c.u[(y / 2) * w.div_ceil(2) + x / 2], c.v[(y / 2) * w.div_ceil(2) + x / 2]);
        assert_eq!(at(25, 40), (outline().0, outline().1, outline().2), "bottom edge is the outline colour");
        assert_eq!(at(14, 25).0, top_face().0, "far face drawn");
        assert_eq!(at(18, 30), (0, 128, 128), "inside untouched");
        assert!(c.y.iter().any(|&y| y == label().0), "the id is drawn");
    }
}
