//! The PP-OCR recognisers (the languages of [`crate::PP_OCR`]) on the GPU (feature `gpu`,
//! `PDFCRAFT_OCR_GPU=1`): the line pieces of a page are prepared exactly as for rten
//! ([`ppocr::prepare`]), batched by width, read by [`PpRecognizer`] with the best classes of each
//! step, and decoded as [`ppocr::decode_ranked`] decodes (the language's letter preference
//! included). With an ONNX detection model the words are found on the GPU too. A self-check
//! against rten decides at load whether the GPU is used; a failure later sends the page and every
//! later one to the CPU.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use pdfcraft_ml::gpu::{Detector, Gpu, PpRecognizer};
use pdfcraft_ml::onnx::Onnx;
use rten_imageproc::RotatedRect;

use crate::Models;
use crate::gpu::{Refused, detector_agrees, word_rects};
use crate::ppocr::{self, PpOcr};

/// Lines read in one batch.
const BATCH: usize = 16;
/// Classes kept per step: enough for the letter preference (every class within its margin of the
/// winner).
const TOP: usize = 8;

pub(crate) struct PpGpu {
    gpu: Gpu,
    rec: PpRecognizer,
    det: Option<Detector>,
    broken: AtomicBool,
    reason: Mutex<Option<String>>,
}

impl PpGpu {
    /// The GPU recogniser for `pp`'s model (and the detector, when ONNX), kept only if they read a
    /// synthetic page and line as rten does.
    pub(crate) fn load(models: &Models, engine: &ocrs::OcrEngine, pp: &PpOcr) -> Result<Self, String> {
        let gpu = Gpu::new().map_err(|e| e.to_string())?;
        let rec = PpRecognizer::new(&gpu, &Onnx::load(&models.recognition).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if rec.classes() != pp.alphabet().len() {
            return Err(format!("the recognition model reads {} classes, the dictionary {}", rec.classes(), pp.alphabet().len()));
        }
        let det = if models.detection.extension().and_then(|e| e.to_str()) == Some("onnx") {
            let det = Detector::new(&gpu, &Onnx::load(&models.detection).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            detector_agrees(&gpu, &det, engine)?;
            Some(det)
        } else {
            None
        };
        let g = Self { gpu, rec, det, broken: AtomicBool::new(false), reason: Mutex::new(None) };
        g.self_check(pp)?;
        Ok(g)
    }

    pub(crate) fn adapter(&self) -> &str {
        self.gpu.adapter()
    }

    /// Whether the GPU is still in use, and why not if it stopped.
    pub(crate) fn status(&self) -> Result<(), String> {
        if self.broken.load(Ordering::Relaxed) {
            return Err(self.reason.lock().ok().and_then(|r| r.clone()).unwrap_or_default());
        }
        Ok(())
    }

    /// Stop using the GPU after `e`.
    pub(crate) fn fail(&self, e: &str) {
        self.broken.store(true, Ordering::Relaxed);
        if let Ok(mut r) = self.reason.lock() {
            r.get_or_insert_with(|| e.to_string());
        }
    }

    /// A synthetic line (dark bars of varying width) read by both: the best class agrees in at
    /// least 49 of 50 steps.
    fn self_check(&self, pp: &PpOcr) -> Result<(), String> {
        let (w, h) = (600usize, 48usize);
        let rgb: Vec<u8> = (0..w * h).flat_map(|i| if ((i % w) / 7 + (i / w) / 9) % 3 == 0 { [20u8; 3] } else { [235u8; 3] }).collect();
        let Some((data, _, width)) =
            ppocr::prepare(&rgb, w as u32, h as u32, [2.0, 2.0, w as f32 - 2.0, h as f32 - 2.0]).map_err(|e| e.to_string())?
        else {
            return Err("the synthetic line is empty".into());
        };
        let classes = pp.alphabet().len();
        let cpu = pp.run(data.clone(), width).map_err(|e| e.to_string())?;
        let cpu_best: Vec<u32> = cpu
            .chunks_exact(classes.max(1))
            .map(|row| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32))
            .collect();
        let gpu = self.rec.recognize(&self.gpu, &[(data, 1, width)], None, 1).map_err(|e| e.to_string())?;
        let got: Vec<u32> = gpu.first().map(|steps| steps.iter().map(|s| s.best().map_or(0, |b| b.0)).collect()).unwrap_or_default();
        let same = got.iter().zip(&cpu_best).filter(|(a, b)| a == b).count();
        if got.len() != cpu_best.len() || same * 50 < cpu_best.len() * 49 {
            return Err(format!("PP-OCR recognition agrees with rten in {same} of {} steps ({} on the GPU)", cpu_best.len(), got.len()));
        }
        Ok(())
    }

    /// The words of an RGB page found on the GPU; `None` when the detection model is not ONNX.
    pub(crate) fn detect(&self, engine: &ocrs::OcrEngine, rgb: &[u8], w: usize, h: usize) -> Result<Option<Vec<RotatedRect>>, Refused> {
        let Some(det) = &self.det else { return Ok(None) };
        let mask = det.detect(&self.gpu, rgb, w, h, engine.detection_threshold())?;
        Ok(Some(word_rects(&mask, w, h)))
    }

    /// The text of each of `rects` (`[left, top, right, bottom]` on the RGB page) in order, as
    /// [`PpOcr::recognize`] reads it with `preferred`.
    pub(crate) fn read(&self, pp: &PpOcr, rgb: &[u8], width: u32, height: u32, rects: &[[f32; 4]], preferred: &str) -> Result<Vec<String>, Refused> {
        let mut prepared: Vec<(usize, Vec<f32>, usize)> = Vec::new();
        for (i, rect) in rects.iter().enumerate() {
            if let Some((data, _, w)) = ppocr::prepare(rgb, width, height, *rect).map_err(|e| Refused::Page(e.to_string()))? {
                prepared.push((i, data, w));
            }
        }
        // Similar widths together, so a batch pads little.
        prepared.sort_by_key(|p| p.2);
        let plane = ppocr::HEIGHT;
        let mut batches: Vec<(Vec<f32>, usize, usize)> = Vec::new();
        let mut members: Vec<(usize, usize, usize)> = Vec::new();
        for chunk in prepared.chunks(BATCH) {
            let bw = chunk.iter().map(|p| p.2).max().unwrap_or(1);
            let n = chunk.len();
            let mut data = vec![0f32; n.saturating_mul(3).saturating_mul(plane).saturating_mul(bw)];
            for (k, (i, src, w)) in chunk.iter().enumerate() {
                for c in 0..3 {
                    for y in 0..plane {
                        let from = (c * plane + y) * w;
                        let to = ((k * 3 + c) * plane + y) * bw;
                        if let (Some(dst), Some(row)) = (data.get_mut(to..to + w), src.get(from..from + w)) {
                            dst.copy_from_slice(row);
                        }
                    }
                }
                members.push((*i, *w, bw));
            }
            batches.push((data, n, bw));
        }
        let read = self.rec.recognize(&self.gpu, &batches, None, TOP)?;
        let mut texts = vec![String::new(); rects.len()];
        for ((i, w, bw), steps) in members.into_iter().zip(read) {
            // Steps past the line's own (padded) width read the batch's padding.
            let t_all = steps.len().max(1);
            let own: Vec<Vec<(u32, f32)>> = steps.into_iter().enumerate().filter(|(t, _)| t * bw / t_all < w).map(|(_, s)| s.ranked).collect();
            if let Some(slot) = texts.get_mut(i) {
                *slot = ppocr::decode_ranked(&own, pp.alphabet(), preferred);
            }
        }
        Ok(texts)
    }
}
