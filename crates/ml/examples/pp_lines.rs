//! Time PP-OCR recognition of the text lines of a page image on the GPU and with rten on the CPU.
//!
//! ```sh
//! cargo run --release -p pdfcraft-ml --features gpu --example pp_lines -- page.png model.onnx [model.onnx …]
//! ```
//!
//! Lines are found by a row profile of dark pixels (a clean, upright page), cut with a margin,
//! and prepared as the PP-OCR pull requests prepare them: resized to 48 pixels high, BGR, scaled
//! to -1..1. rten reads them one at a time (width at least 320, as there); the GPU in batches of
//! up to 16 lines sorted by width. Prints milliseconds per line and how many steps agree.

use std::time::Instant;

use image::imageops::{self, FilterType};
use pdfcraft_ml::gpu::{Gpu, PpRecognizer};
use pdfcraft_ml::onnx::Onnx;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};

const H: usize = 48;
const BATCH: usize = 16;

/// The [3, 48, width] input of a line image, padded with zeros to `width`.
fn prepare(img: &image::RgbImage, width: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; 3 * H * width];
    for (x, y, p) in img.enumerate_pixels() {
        for c in 0..3 {
            if let Some(v) = data.get_mut((c * H + y as usize) * width + x as usize) {
                *v = p[2 - c] as f32 / 127.5 - 1.0;
            }
        }
    }
    data
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (page, models) = args.split_first().ok_or("usage: pp_lines page.png model.onnx …")?;
    let page = image::open(page)?.to_rgb8();
    let (w, h) = page.dimensions();
    // Rows with ink, grouped into lines.
    let dark = |x: u32, y: u32| page.get_pixel(x, y).0.iter().map(|&v| v as u32).sum::<u32>() < 384;
    let inked: Vec<bool> = (0..h).map(|y| (0..w).filter(|&x| dark(x, y)).count() > 2).collect();
    let mut lines = Vec::new();
    let mut y = 0;
    while y < h {
        if inked[y as usize] {
            let top = y;
            while y < h && inked[y as usize] {
                y += 1;
            }
            let (left, right) = (0..w).filter(|&x| (top..y).any(|yy| dark(x, yy))).fold((w, 0), |(l, r), x| (l.min(x), r.max(x)));
            if y - top >= 8 && right > left {
                let pad = (y - top) / 4;
                let (x0, y0) = (left.saturating_sub(pad), top.saturating_sub(pad));
                let (x1, y1) = ((right + pad).min(w), (y + pad).min(h));
                let crop = imageops::crop_imm(&page, x0, y0, x1 - x0, y1 - y0).to_image();
                let rw = ((H as f64 * crop.width() as f64 / crop.height() as f64).ceil() as u32).clamp(1, 3200);
                lines.push(imageops::resize(&crop, rw, H as u32, FilterType::Triangle));
            }
        }
        y += 1;
    }
    println!("{} lines", lines.len());
    let gpu = Gpu::new()?;
    println!("{}", gpu.adapter());
    for path in models {
        let onnx = Onnx::load(std::path::Path::new(path))?;
        let rec = PpRecognizer::new(&gpu, &onnx)?;
        let model = rten::Model::load_file(path)?;
        // rten, a line at a time.
        let t = Instant::now();
        let mut cpu = Vec::new();
        for img in &lines {
            let width = (img.width() as usize).max(320);
            let out: Tensor<f32> = model.run_one(NdTensor::from_data([1, 3, H, width], prepare(img, width)).view().into(), None)?.try_into()?;
            let [_, steps, classes] = <[usize; 3]>::try_from(out.shape().to_vec()).map_err(|_| "output shape")?;
            let row = |t: usize| {
                (0..classes).max_by(|&a, &b| out.get([0, t, a]).unwrap_or(&0.0).total_cmp(out.get([0, t, b]).unwrap_or(&0.0))).unwrap_or(0)
            };
            cpu.push((0..steps).map(row).collect::<Vec<_>>());
        }
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3;
        // The GPU, in batches sorted by width; a warm-up run first.
        let mut order: Vec<usize> = (0..lines.len()).collect();
        order.sort_by_key(|&i| lines[i].width());
        let batches: Vec<(Vec<f32>, usize, usize)> = order
            .chunks(BATCH)
            .map(|chunk| {
                let width = chunk.iter().map(|&i| lines[i].width() as usize).max().unwrap_or(1).max(320).next_multiple_of(8);
                (chunk.iter().flat_map(|&i| prepare(&lines[i], width)).collect(), chunk.len(), width)
            })
            .collect();
        rec.recognize(&gpu, &batches, None, 1)?;
        let t = Instant::now();
        let got = rec.recognize(&gpu, &batches, None, 1)?;
        let gpu_ms = t.elapsed().as_secs_f64() * 1e3;
        // Steps that read the line itself (not the batch's padding past it, nor rten's past 320).
        let (mut same, mut total) = (0, 0);
        for (k, &i) in order.iter().enumerate() {
            let (Some(g), Some(c)) = (got.get(k), cpu.get(i)) else { continue };
            let batch_w = batches.get(k / BATCH).map_or(1, |b| b.2);
            let own = (lines[i].width() as usize * g.len()).div_ceil(batch_w);
            for (t, step) in g.iter().take(own.min(c.len())).enumerate() {
                total += 1;
                same += usize::from(step.best().map(|b| b.0 as usize) == c.get(t).copied());
            }
        }
        let n = lines.len().max(1) as f64;
        println!(
            "{}: {} classes; rten {:.1} ms a line, GPU {:.2} ms a line ({:.1}x); {same} of {total} steps agree",
            path,
            rec.classes(),
            cpu_ms / n,
            gpu_ms / n,
            cpu_ms / gpu_ms.max(1e-9)
        );
    }
    Ok(())
}
