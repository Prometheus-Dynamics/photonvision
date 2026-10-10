//! The processed stream: the frame's luma with each detected tag's outline drawn, as a grey JPEG.

/// Draw the outline (four edges) of each quad, given as `[u0, v0, ..., u3, v3]`, into `luma`
/// (`width` x `height`, rows `stride` bytes apart) with value `value`.
pub fn draw_quads(luma: &mut [u8], width: usize, height: usize, stride: usize, quads: &[[f64; 8]], value: u8) {
    for quad in quads {
        for edge in 0..4 {
            let a = (quad[edge * 2], quad[edge * 2 + 1]);
            let next = (edge + 1) % 4;
            let b = (quad[next * 2], quad[next * 2 + 1]);
            draw_line(luma, width, height, stride, a, b, value);
        }
    }
}

fn draw_line(luma: &mut [u8], width: usize, height: usize, stride: usize, a: (f64, f64), b: (f64, f64), value: u8) {
    if !(a.0.is_finite() && a.1.is_finite() && b.0.is_finite() && b.1.is_finite()) {
        return;
    }
    let steps = (b.0 - a.0).abs().max((b.1 - a.1).abs()).ceil().max(1.0) as usize;
    for s in 0..=steps {
        let t = s as f64 / steps as f64;
        let x = (a.0 + (b.0 - a.0) * t).round();
        let y = (a.1 + (b.1 - a.1) * t).round();
        if x < 0.0 || y < 0.0 || x >= width as f64 || y >= height as f64 {
            continue;
        }
        luma[y as usize * stride + x as usize] = value;
    }
}

/// Encode a grey image (`width` x `height`, rows `stride` bytes apart) as a JPEG.
pub fn encode_grey_jpeg(luma: &[u8], width: usize, height: usize, stride: usize, quality: u8) -> Result<Vec<u8>, String> {
    let mut packed = Vec::with_capacity(width * height);
    for row in 0..height {
        packed.extend_from_slice(&luma[row * stride..row * stride + width]);
    }
    let mut out = Vec::with_capacity(width * height / 4);
    let encoder = jpeg_encoder::Encoder::new(&mut out, quality);
    encoder
        .encode(&packed, width as u16, height as u16, jpeg_encoder::ColorType::Luma)
        .map_err(|e| format!("jpeg encode failed: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outlines_a_square_and_encodes_a_jpeg() {
        let (w, h) = (64usize, 48usize);
        let mut luma = vec![0u8; w * h];
        let quad = [10.0, 10.0, 40.0, 10.0, 40.0, 30.0, 10.0, 30.0];
        draw_quads(&mut luma, w, h, w, &[quad], 255);
        assert_eq!(luma[10 * w + 10], 255);
        assert_eq!(luma[10 * w + 25], 255);
        assert_eq!(luma[20 * w + 25], 0, "interior stays untouched");
        let jpeg = encode_grey_jpeg(&luma, w, h, w, 80).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
    }
}
