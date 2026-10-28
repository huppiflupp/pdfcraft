# Local text recognition

Scan & OCR → Recognize text offers **English** (`en`), **Chinese (Simplified) and English**
(`zh`), **Latin (accented)** (`la`), and the individual languages French, German, Spanish,
Portuguese, Italian, Dutch, Danish, Norwegian, Swedish, Icelandic, Finnish, Czech and Polish.
The Chinese recogniser reads Chinese, Latin letters and numbers in the same line. `la` and
those named languages share the Latin recogniser. A named language prefers its own letters
when two readings are close (`å` rather than `à` for Danish) and still keeps a letter from
another language when that reading is clearly ahead. The dictionary in
[ATTRIBUTION.toml](../ATTRIBUTION.toml) lists the exact character set. Choose the document
language in the dialog before recognising the current page, a range, all pages or multiple
files. The interface language and OCR document language are independent.

The output is a searchable image: original page images and content stay intact, with an
invisible text layer for selection, copying, search and text export. Recognition is one
undoable step. By default, pages that already contain text are skipped.

## Install the models

From a source checkout:

```sh
cargo xtask models
```

This downloads the models and licence notices into `assets/models/`, verifies their pinned
SHA-256 hashes, and leaves them outside Git. English needs `text-detection.rten` and
`text-recognition.rten`; Chinese needs `text-detection.rten`, `chinese-recognition.onnx`
and `chinese-keys.txt`; Latin needs `text-detection.rten`, `latin-recognition.onnx`
and `latin-keys.txt`. Sources, versions, hashes and licences are in
[ATTRIBUTION.toml](../ATTRIBUTION.toml).

For an installed desktop app, put those files in a directory and launch with
`PDFCRAFT_MODELS` pointing to its absolute path. Alternatively, place `models/` beside
the executable, or in `PdfCraft.app/Contents/Resources/` on macOS. The source tree's
`assets/models/` is the final fallback for development builds. Model availability is
checked separately for each language; installing English alone does not enable Chinese
or Latin. Desktop release packages include the models. The Linux CLI-only archive does not.

Recognition runs entirely on the CPU in Rust, using RTen. English uses ocrs;
Chinese uses the Apache-2.0 PP-OCRv4 mobile recognition model and Latin the PP-OCRv5
Latin one, both converted to ONNX by RapidAI, with PaddleOCR's character dictionaries and
the existing ocrs detector. Only model installation needs network access. Recognition
never uploads a PDF or calls a cloud service.

## Automation

`ocr_recognize` and `ocr_recognize_files` accept `"language": "zh"`, `"language": "la"`, or a
named language such as `"fr"` or `"es"`.
`ocr_status` lists each language and its `available` flag. The legacy top-level `available`
flag continues to report English availability. For example, after `doc_open` returns
document 1:

```json
{"tool":"ocr_recognize","args":{"doc":1,"language":"zh","pages":[1]}}
```

## Ground truth and benchmark

`crates/engine/tests/data/ocr/` holds ground-truth texts: `de.tsv` (an invoice, a letter and a
delivery note written for PdfCraft, with amounts, dates, `§`, `°C` and dashes) and, for each
named Latin language, three paragraphs from that language's Wikipedia (CC BY-SA 4.0, revision
recorded per row). `cargo xtask ocr-texts <lang>` fetches new ones; it picks the paragraphs
whose letters are most often outside ASCII and prints the `ATTRIBUTION.toml` entry. The tests
never touch the network.

`tests/it/support/ocr_scan.rs` sets each text on an A4 page in a committed face (Inter,
JetBrains Mono) and scans it under four conditions: 300 dpi clean; 300 dpi with noise, blur and
a 1° skew; 200 dpi with noise; 150 dpi at JPEG quality 65. Noise is seeded, so every run sees the
same pixels. Words are scored as a multiset (reading order does not count), in three classes:
all words, words with a letter outside ASCII, and tokens with a digit or symbol.

`ocr_ground_truth::scanned_ground_truth_reads_back_above_its_floor` checks every fixture in
Inter at 200 dpi against a recorded floor. The full matrix is

```sh
cargo run --release -p pdfcraft-engine --example ocr_bench -- --all --misses
```

Measured 2026-10-26 (both faces, all four conditions, 24 pages per row; `en` is the ocrs
recogniser on `main`):

| Fixture | `en` words | Language words | `en` accented | Language accented |
|---|---|---|---|---|
| de (invoice, letter, delivery note) | 65.7 % | 97.4 % | 0 % | 99.4 % |
| de (Wikipedia) | 73.3 % | 100 % | 0 % | 100 % |
| fr | 68.8 % | 99.9 % | 0 % | 100 % |
| es | 73.2 % | 99.8 % | 0 % | 99.0 % |
| pt | 69.0 % | 100 % | 0 % | 100 % |
| it | 80.4 % | 99.5 % | 0 % | 99.3 % |
| nl | 85.2 % | 99.4 % | 0 % | 91.2 % |
| fi | 64.0 % | 97.8 % | 0 % | 98.5 % |
| cs | 35.1 % | 87.6 % | 0 % | 78.4 % |
| sv | 50.8 % | 87.6 % | 0 % | 72.0 % |
| da | 74.1 % | 94.3 % | 0 % | 68.3 % |
| is | 37.6 % | 79.4 % | 0 % | 61.9 % |
| no | 73.2 % | 93.8 % | 0 % | 55.7 % |
| pl | 46.2 % | 73.6 % | 0 % | 35.9 % |

In the German documents the words still missed are almost all typography: the minus sign `−`
and the en dash `–` come back as `-`, and `·` as `.`.

### Real pages: MDPBench

The fixtures above are synthetic. For real pages there is
[MDPBench](https://huggingface.co/datasets/Delores-Lin/MDPBench)'s public split (Apache-2.0). It
has 160 pages per language, among them German, English, Spanish, French, Italian, Dutch and
Portuguese: academic papers, newspapers, books, magazines, financial reports and letters. For
each digital original there are three phone photos with folds, shadows, glare and blur.
MDPBench is not committed and no test reads it.

```sh
cargo xtask mdpbench --langs de,fr,es        # once; pinned revision, into target/mdpbench/
cargo run --release -p pdfcraft-engine --example ocr_mdpbench -- target/mdpbench --langs de,fr,es --ocr en,la --misses
```

Pages are scored in the same three word classes against the text of their annotated regions.
Table cells, taken from the annotated HTML, are scored in a column of their own, so misread
figures in tables show. Formulas are not scored, because recognised words cannot be compared
with LaTeX. The results are split into three groups:

- `digital`: the originals.
- `photo`: photos taken from above.
- `photo-turned`: photos taken from the left, the right or below. Their text lies sideways or
  upside down, and recognition cannot read it without turning the page first.

The resolution is not recorded, so each page is read as if it were A4 wide.

## Limits

This is an initial searchable-image workflow, not an editable-text reconstruction.
It targets horizontal printed text. Handwriting, vertical text, automatic deskew, reliable
table reconstruction and rotated-line recognition remain unsupported. Reading order uses
the existing detector's line analysis; complex layouts can need review. Chinese and Latin
selection boxes cover a recognised line, with approximate character positions within it.
The Latin recogniser reads only the characters in its dictionary: accented letters outside
it (e.g. some East European ones) are still dropped. A letter that is in the dictionary can
still come back as another letter in it: on Árvíztűrő tükörfúrógép the model read ű as ü.
The same is true, by more than the close-call margin that `preferred_letters` can bridge, of
Danish, Norwegian, Swedish and Finnish lowercase å (misread as à, á or ã), Icelandic þ (misread
as a plain "b"), Czech's comma-shaped mark on ď, ť and ň and its ring on ů (all dropped), and
Polish ą, ź and ł (dropped) and ż (misread as Czech's ž). Each language's other accented
letters, including its vowels, come back correctly; `crates/engine/src/tests.rs`'s
`latin_ocr_keeps_*` tests cover the letters that do and name the ones that do not, language by
language.

OCR does not make visible text in a scan directly editable. Existing selectable PDF text
uses Edit PDF → Edit text and images, whose font/subset and CJK limitations are separate.
Vector artwork and nested image objects are also separate from OCR.

The core round-trip tests generate their own scans: a Chinese/English one using the permitted
BIZ UDPGothic font from `CRAFT_FONTS_DIR`, and a German one using the committed Inter face,
with umlauts and `ß` that the English models drop as `?`. Both require the model files.
Unicode-layer tests run without models or fonts and verify save/reopen, undo, supplementary
characters, font code boundaries and unchanged rendered pixels.
