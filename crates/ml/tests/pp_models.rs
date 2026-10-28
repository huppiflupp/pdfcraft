//! PP-OCR recognition on the GPU against rten on the CPU, for each model found: the best class
//! of every step, with and without a class mask, and its probability.
//!
//! The models: `PDFCRAFT_PP_MODELS` (`.onnx` files, separated as in `PATH`), else in
//! `PDFCRAFT_MODELS` the names the PP-OCR pull requests install (`multilingual-recognition.onnx`,
//! PP-OCRv6; `latin-recognition.onnx`, PP-OCRv5 Latin) and `ppocr/PP-OCRv6_small_rec.onnx`
//! (wgpu-ocr's layout). Skips without a GPU or a model.

#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use std::path::PathBuf;

use pdfcraft_ml::gpu::{Gpu, PpRecognizer};
use pdfcraft_ml::onnx::Onnx;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};

fn models() -> Vec<PathBuf> {
    if let Some(list) = std::env::var_os("PDFCRAFT_PP_MODELS") {
        return std::env::split_paths(&list).filter(|p| p.is_file()).collect();
    }
    let Some(dir) = std::env::var_os("PDFCRAFT_MODELS").map(PathBuf::from) else { return Vec::new() };
    ["multilingual-recognition.onnx", "latin-recognition.onnx", "ppocr/PP-OCRv6_small_rec.onnx"]
        .iter()
        .map(|f| dir.join(f))
        .filter(|p| p.is_file())
        .collect()
}

/// Two lines of letter-like strokes on light paper, [2, 3, 48, width], in the models' -1..1.
fn lines(width: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; 2 * 3 * 48 * width];
    for line in 0..2 {
        for c in 0..3 {
            for y in 0..48 {
                for x in 0..width {
                    let glyph = x / 14 + line * 3;
                    let (gx, gy) = (x % 14, y);
                    let ink = (8..40).contains(&gy)
                        && match glyph % 5 {
                            0 => !(3..=9).contains(&gx),
                            1 => gx < 3 || gy < 11 || (22..25).contains(&gy),
                            2 => (gx + gy / 3) % 10 < 3,
                            3 => gx < 3 || gy > 36,
                            _ => gx == 6 || gx == 7 || gy < 11,
                        }
                        && x < width - 30 * line;
                    let v = if ink { 30.0 } else { 235.0 };
                    data[((line * 3 + c) * 48 + y) * width + x] = v / 127.5 - 1.0;
                }
            }
        }
    }
    data
}

#[test]
fn pp_ocr_on_the_gpu_reads_steps_as_rten_does() {
    let paths = models();
    if paths.is_empty() {
        eprintln!("skipped: no PP-OCR recognition model (PDFCRAFT_PP_MODELS or PDFCRAFT_MODELS)");
        return;
    }
    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipped: {e}");
            return;
        }
    };
    let width = 320;
    let data = lines(width);
    for path in paths {
        let onnx = Onnx::load(&path).unwrap();
        let rec = PpRecognizer::new(&gpu, &onnx).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let model = rten::Model::load_file(&path).unwrap();
        let out: Tensor<f32> = model.run_one(NdTensor::from_data([2, 3, 48, width], data.clone()).view().into(), None).unwrap().try_into().unwrap();
        let [n, steps, classes] = <[usize; 3]>::try_from(out.shape().to_vec()).unwrap();
        assert_eq!((n, classes), (2, rec.classes()), "{}", path.display());
        let prob = |line: usize, t: usize, c: usize| out.get([line, t, c]).copied().unwrap();
        let allowed: Vec<bool> = (0..classes).map(|c| c % 2 == 0).collect();
        for mask in [None, Some(&allowed[..])] {
            let got = rec.recognize(&gpu, &[(data.clone(), 2, width)], mask, 3).unwrap();
            assert_eq!(got.len(), 2);
            let (mut same, mut total) = (0, 0);
            for (line, line_steps) in got.iter().enumerate() {
                assert_eq!(line_steps.len(), steps);
                for (t, step) in line_steps.iter().enumerate() {
                    let want = (0..classes)
                        .filter(|&c| mask.is_none() || allowed[c])
                        .max_by(|&a, &b| prob(line, t, a).total_cmp(&prob(line, t, b)))
                        .unwrap();
                    let (best, p) = step.best().unwrap();
                    assert!(mask.is_none() || allowed[best as usize], "a masked class {best} came back");
                    total += 1;
                    if best as usize == want {
                        same += 1;
                        assert!((p - prob(line, t, want)).abs() < 2e-3, "{}: step {t}: {p} against {}", path.display(), prob(line, t, want));
                    }
                }
            }
            eprintln!("{}: {same} of {total} steps as rten (mask {})", path.display(), mask.is_some());
            assert!(same * 50 >= total * 49, "{}: {same} of {total} steps as rten (mask {})", path.display(), mask.is_some());
        }
    }
}

#[test]
fn malformed_batches_and_masks_are_errors() {
    let Some(path) = models().into_iter().next() else {
        eprintln!("skipped: no PP-OCR recognition model");
        return;
    };
    let Ok(gpu) = Gpu::new() else {
        eprintln!("skipped: no GPU");
        return;
    };
    let rec = PpRecognizer::new(&gpu, &Onnx::load(&path).unwrap()).unwrap();
    let good = vec![0.0f32; 3 * 48 * 64];
    assert!(rec.recognize(&gpu, &[(good.clone(), 2, 64)], None, 1).is_err(), "too few values for two lines");
    assert!(rec.recognize(&gpu, &[(Vec::new(), 0, 64)], None, 1).is_err(), "no lines");
    assert!(rec.recognize(&gpu, &[(good.clone(), 1, 64)], Some(&[true; 3]), 1).is_err(), "a mask of the wrong length");
    assert!(rec.recognize(&gpu, &[(good.clone(), 1, 64)], None, 0).is_err(), "no classes a step");
    assert!(rec.recognize(&gpu, &[(good, 1, 64)], None, 9).is_err(), "more than TOP classes a step");
    assert!(rec.recognize(&gpu, &[], None, 1).is_ok_and(|l| l.is_empty()));
}
