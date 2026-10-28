# pdfcraft-ml

Machine-learning inference on the GPU (layer 3). With the `gpu` feature (native targets only),
the OCR models run in wgpu compute shaders (Vulkan, Metal, DX12), read from the same `.onnx`
files the CPU path (rten) loads. Without the feature the crate is empty.

- `onnx`: the graph and weights, read with `rten-onnx` (rten's parser, already in the tree).
- `gpu::Gpu`: the device. Work runs in error scopes; validation and out-of-memory errors, uncaptured
  errors and a lost device come back as `GpuError`, never a panic. Sizes from the model file are
  checked against the device's limits. Software rasterisers (llvmpipe) are refused: they would
  be slower than rten.
- `gpu::Detector`, `gpu::Recognizer`: the ocrs detection and recognition models (convolutional
  graph executor; bidirectional GRU in one dispatch per layer, each line with its own length;
  recurrent weights in f16 where `SHADER_F16` is available).
- `gpu::PpRecognizer`: a PP-OCR recognition model (PP-OCRv6, the PP-OCRv5 Latin model) on an
  N-dimensional graph executor (`gpu/nd.rs`: the CNN backbone, the SVTR neck with LayerNorm and
  attention, the CTC head; bias, activations, residuals and squeeze-excitation fused into the
  convolutions). The final Softmax and the ranking of each step's classes run in one kernel, so
  only the best few classes per step are read back; an optional class mask keeps classes out
  (e.g. a Latin-only decode). The caller prepares the line images and decodes the steps.
  `examples/pp_lines.rs` times it against rten on a page's lines; `tests/pp_models.rs` checks it
  against rten (`PDFCRAFT_PP_MODELS` or `PDFCRAFT_MODELS`, skipped without a model or a GPU).

Scratch buffers come from a pool. A graph frees each value after its last reader while it is
being recorded, and the pool, when full, drops other sizes first, so a batch holds about its
live values rather than every value it made.

`pdfcraft-ocr` uses it behind its own `gpu` feature, opt-in at run time with
`PDFCRAFT_OCR_GPU=1`, and keeps the CPU path as the fallback (see that crate).

wgpu is otherwise a UI dependency (L7+): `xtask/src/layers.rs` allows it here for compute only
(`GPU_COMPUTE`), and only wgpu.

Measurements, the kernels' history and a comparison with Tesseract and PaddleOCR are in
the prototype this came from: <https://github.com/huppiflupp/wgpu-ocr>.
