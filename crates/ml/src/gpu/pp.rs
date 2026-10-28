//! A PP-OCR recognition model (PP-OCRv6, the PP-OCRv5 Latin model) on the GPU: the graph up to
//! its final Softmax ([`super::nd`]), then the Softmax and a ranking of each step's classes in one
//! kernel, so only the best few classes of each step are read back, not the logits.
//!
//! The caller prepares the input as the CPU path does (line crops resized to [`HEIGHT`] pixels,
//! BGR, scaled to -1..1, padded on the right) and decodes the steps (CTC): which classes it may
//! use ([`PpRecognizer::recognize`]'s mask), and how close a preferred letter has to be, are its
//! choices.

use super::graph::count;
use super::nd::{NdGraph, Tensor};
use super::{Buf, Gpu, GpuError, params};
use crate::onnx::Onnx;

/// The models' input height.
pub const HEIGHT: usize = 48;
/// The most classes a step reports (the ranking kernel's list length).
pub const TOP: usize = 8;

/// One step of a line: its classes with the highest scores among those allowed, best first, each
/// with its probability (the softmax over every class, allowed or not).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Step {
    pub ranked: Vec<(u32, f32)>,
}

impl Step {
    /// The best allowed class and its probability.
    pub fn best(&self) -> Option<(u32, f32)> {
        self.ranked.first().copied()
    }
}

pub struct PpRecognizer {
    graph: NdGraph,
    classes: usize,
}

impl PpRecognizer {
    pub fn new(gpu: &Gpu, onnx: &Onnx) -> Result<Self, GpuError> {
        let graph = gpu.scoped(|| NdGraph::new(gpu, onnx))?;
        let classes = graph.classes().ok_or_else(|| GpuError::Model("the classes are not a constant linear layer before the Softmax".into()))?;
        if classes < 2 {
            return Err(GpuError::Model(format!("{classes} classes")));
        }
        Ok(Self { graph, classes })
    }

    /// The classes the model reads: the CTC blank (0), then its dictionary.
    pub fn classes(&self) -> usize {
        self.classes
    }

    pub fn height(&self) -> usize {
        HEIGHT
    }

    /// Recognise `batches` of lines, each `(data, lines, width)` with `data` the [lines, 3,
    /// HEIGHT, width] input. Returns, per line in order (batch by batch), the model's steps for
    /// its batch's width: step `t` of `T` reads input columns `t * width / T` up to `(t + 1) *
    /// width / T`, so the steps past a line's own width read the padding (the caller drops
    /// them). `allowed`, one flag per class, keeps the others out of the ranking; `top` (1 to
    /// [`TOP`]) classes are kept per step.
    pub fn recognize(
        &self,
        gpu: &Gpu,
        batches: &[(Vec<f32>, usize, usize)],
        allowed: Option<&[bool]>,
        top: usize,
    ) -> Result<Vec<Vec<Step>>, GpuError> {
        if top == 0 || top > TOP {
            return Err(GpuError::Model(format!("{top} classes a step; 1 to {TOP} are kept")));
        }
        if allowed.is_some_and(|a| a.len() != self.classes) {
            return Err(GpuError::Model(format!("a mask of {} flags for {} classes", allowed.map_or(0, <[bool]>::len), self.classes)));
        }
        for (data, n, w) in batches {
            if *n == 0 || *w == 0 || Some(data.len()) != count(&[*n, 3, HEIGHT, *w]).ok() {
                return Err(GpuError::Model(format!("a batch of {} values for {n} lines of width {w}", data.len())));
            }
        }
        let mut words = vec![0u32; self.classes.div_ceil(32)];
        if let Some(a) = allowed {
            for (c, _) in a.iter().enumerate().filter(|(_, ok)| **ok) {
                if let Some(word) = words.get_mut(c / 32) {
                    *word |= 1 << (c % 32);
                }
            }
        }
        gpu.scoped(|| {
            let mask = gpu.upload_u32(&words)?;
            let mut lines = Vec::new();
            for (data, n, w) in batches {
                lines.extend(self.batch(gpu, data, *n, *w, &mask, allowed.is_some(), top)?);
            }
            Ok(lines)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn batch(&self, gpu: &Gpu, data: &[f32], n: usize, w: usize, mask: &Buf, masked: bool, top: usize) -> Result<Vec<Vec<Step>>, GpuError> {
        let mut scratch = Vec::new();
        let mut enc = gpu.encoder();
        let input = gpu.upload(data)?;
        let logits = self.graph.run(gpu, &mut enc, Tensor { buf: input, shape: vec![n, 3, HEIGHT, w] });
        let result = logits.and_then(|logits| {
            scratch.push(logits.buf.clone());
            let steps = match logits.shape[..] {
                [ln, t, c] if ln == n && c == self.classes && t > 0 => t,
                _ => return Err(GpuError::Model(format!("logits {:?} for {n} lines of {} classes", logits.shape, self.classes))),
            };
            let rows = count(&[n, steps])?;
            let raw = rank(gpu, enc, &logits.buf, rows, self.classes, mask, masked, &mut scratch)?;
            Ok(ranked(&raw, n, steps, self.classes, top))
        });
        gpu.recycle(scratch);
        result
    }
}

/// Record the final Softmax and the ranking over `rows` rows of `classes` logits, submit `enc`
/// and read the ranking back (per row TOP classes, then TOP probability bits).
#[allow(clippy::too_many_arguments)]
fn rank(
    gpu: &Gpu,
    mut enc: wgpu::CommandEncoder,
    logits: &Buf,
    rows: usize,
    classes: usize,
    mask: &Buf,
    masked: bool,
    scratch: &mut Vec<Buf>,
) -> Result<Vec<u32>, GpuError> {
    let out = gpu.storage(count(&[rows, 2 * TOP])?)?;
    scratch.push(out.clone());
    let p = gpu.uniform(&params(&[rows, classes, usize::from(masked), 0])?)?;
    gpu.rows(&mut enc, "softmax_topk", &[logits, mask, &out, &p], rows)?;
    gpu.read(enc, &out, count(&[rows, 2 * TOP])?)
}

/// The kernel's output (per row TOP classes, then TOP probability bits) as `n` lines of `steps`
/// steps, each with at most `top` classes; empty slots and out-of-range classes are dropped.
fn ranked(raw: &[u32], n: usize, steps: usize, classes: usize, top: usize) -> Vec<Vec<Step>> {
    let mut rows = raw.as_chunks::<{ 2 * TOP }>().0.iter().map(|row| {
        let (ids, probs) = row.split_at(TOP);
        Step { ranked: ids.iter().zip(probs).filter(|(c, _)| (**c as usize) < classes).take(top).map(|(&c, &p)| (c, f32::from_bits(p))).collect() }
    });
    (0..n).map(|_| rows.by_ref().take(steps).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ranking_is_read_line_by_line() {
        // Two lines of one step each, 5 classes: the first row ranks 3 then 1, the second
        // only 4 (the rest empty); a class out of range is dropped.
        let none = u32::MAX;
        let mut raw = vec![3, 1, 7, none, none, none, none, none];
        raw.extend([0.6f32, 0.3, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0].map(f32::to_bits));
        raw.extend([4, none, none, none, none, none, none, none]);
        raw.extend([0.9f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].map(f32::to_bits));
        let lines = ranked(&raw, 2, 1, 5, 8);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0][0].ranked, vec![(3, 0.6), (1, 0.3)]);
        assert_eq!(lines[1][0].best(), Some((4, 0.9)));
        assert_eq!(ranked(&raw, 2, 1, 5, 1)[0][0].ranked, vec![(3, 0.6)]);
        // Fewer values than lines need: short lines, no panic.
        assert!(ranked(&raw[..16], 2, 1, 5, 8)[1].is_empty());
    }

    /// The ranking kernel against the CPU on synthetic logits: the softmax over every class, the
    /// best allowed classes in order, ties to the lower class. Needs a GPU; skips without one.
    #[test]
    fn the_kernel_ranks_the_allowed_classes_by_their_softmax() {
        let gpu = match Gpu::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipped: {e}");
                return;
            }
        };
        let (rows, classes): (usize, usize) = (37, 1000);
        // Deterministic logits with repeats (ties) and a few large values.
        let logits: Vec<f32> =
            (0..rows * classes).map(|i| (((i * 7919) % 613) as f32 / 61.3).sin() * 4.0 + if i % 97 == 0 { 9.0 } else { 0.0 }).collect();
        let allowed: Vec<bool> = (0..classes).map(|c| c % 3 != 1).collect();
        for masked in [false, true] {
            let mut words = vec![0u32; classes.div_ceil(32)];
            for c in (0..classes).filter(|&c| allowed[c]) {
                words[c / 32] |= 1 << (c % 32);
            }
            let raw = gpu
                .scoped(|| {
                    let (buf, mask) = (gpu.upload(&logits)?, gpu.upload_u32(&words)?);
                    rank(&gpu, gpu.encoder(), &buf, rows, classes, &mask, masked, &mut Vec::new())
                })
                .unwrap();
            let got = ranked(&raw, rows, 1, classes, TOP);
            for (r, line) in got.iter().enumerate() {
                let row = &logits[r * classes..(r + 1) * classes];
                let top = row.iter().copied().fold(f32::MIN, f32::max);
                let sum: f32 = row.iter().map(|v| (v - top).exp()).sum();
                let mut want: Vec<usize> = (0..classes).filter(|&c| !masked || allowed[c]).collect();
                want.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
                let step = &line[0];
                assert_eq!(step.ranked.len(), TOP);
                for (k, &(c, p)) in step.ranked.iter().enumerate() {
                    assert_eq!(c as usize, want[k], "row {r} rank {k} masked {masked}");
                    let expect = (row[c as usize] - top).exp() / sum;
                    assert!((p - expect).abs() < 1e-5, "row {r} class {c}: {p} against {expect}");
                }
            }
        }
    }
}
