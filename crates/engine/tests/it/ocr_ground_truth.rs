//! #524: every OCR ground-truth text (`tests/data/ocr/`), set in Inter and scanned at 200 dpi with
//! noise, blur and JPEG compression, reads back at least as well as it did when the floor was
//! recorded. The floors are the measured share minus a margin, so a regression fails while the
//! known weak letters (Nordic å, Icelandic þ/ð, Polish ą/ł/ź, see docs/ocr.md) do not.
//! Skipped when the Latin models are not installed (`cargo xtask models`). The full matrix of
//! faces and scan conditions is `cargo run --release -p pdfcraft-engine --example ocr_bench -- --all`.

use pdfcraft_engine::{ImageResolution, Session, ocr};

use crate::support::ocr_scan::{self, Score};

/// (fixture file, minimum share of all words, minimum share of words with a letter outside ASCII).
/// Recorded 2026-10-26 on PP-OCRv5 Latin with the language's preferred letters.
const FLOORS: &[(&str, f64, f64)] = &[
    ("cs-wikipedia.tsv", 0.88, 0.82),
    ("da-wikipedia.tsv", 0.88, 0.65),
    ("de-wikipedia.tsv", 0.95, 0.95),
    ("de.tsv", 0.92, 0.95),
    ("es-wikipedia.tsv", 0.95, 0.95),
    ("fi-wikipedia.tsv", 0.93, 0.92),
    ("fr-wikipedia.tsv", 0.95, 0.95),
    ("is-wikipedia.tsv", 0.72, 0.52),
    ("it-wikipedia.tsv", 0.95, 0.95),
    ("nl-wikipedia.tsv", 0.93, 0.80),
    ("no-wikipedia.tsv", 0.87, 0.48),
    ("pl-wikipedia.tsv", 0.66, 0.26),
    ("pt-wikipedia.tsv", 0.95, 0.95),
    ("sv-wikipedia.tsv", 0.80, 0.65),
];

#[test]
fn every_fixture_has_a_floor_and_inter_draws_it() {
    let faces = ocr_scan::builtin_faces();
    for f in ocr_scan::all() {
        assert!(FLOORS.iter().any(|(file, ..)| *file == f.file), "{}: add a floor to FLOORS", f.file);
        assert!(ocr::LANGUAGES.iter().any(|(code, _)| *code == f.lang), "{}: {} is not an OCR language", f.file, f.lang);
        for doc in &f.docs {
            assert_eq!(ocr_scan::missing_glyphs(&faces[0], &doc.text()), Vec::<char>::new(), "{} {}", f.file, doc.name);
        }
    }
}

#[test]
fn scanned_ground_truth_reads_back_above_its_floor() {
    let face = &ocr_scan::builtin_faces()[0];
    let cond = ocr_scan::condition("200noisy").unwrap();
    let mut failures = Vec::new();
    for f in ocr_scan::all() {
        if !ocr::available_for(&f.lang) {
            eprintln!("skipped: OCR models for {} not installed", f.lang);
            return;
        }
        let mut total = Score::default();
        for doc in &f.docs {
            let image = ocr_scan::scan(doc, face, cond, ocr_scan::seed(&[&f.file, &doc.name, &face.name, cond.name])).unwrap();
            let mut s = Session::new();
            let pdf = s.create_from_images_with_resolution(&[("scan.jpg".into(), image)], ImageResolution::Dpi(f64::from(cond.dpi))).unwrap();
            let id = s.open("scan.pdf", None, pdf, None).unwrap();
            let settings = ocr::OcrSettings { dpi: cond.dpi as f32, language: f.lang.clone(), ..Default::default() };
            let text: Vec<String> = s.recognize_text(id, &[], settings).unwrap().iter().map(|p| p.text()).collect();
            total.add(ocr_scan::score(&doc.text(), &text.join("\n")));
        }
        let Some((_, words, accented)) = FLOORS.iter().find(|(file, ..)| *file == f.file) else { continue };
        if total.words_rate() < *words || total.accented_rate() < *accented {
            failures.push(format!(
                "{}: words {:.3} (floor {words}), accented {:.3} (floor {accented})",
                f.file,
                total.words_rate(),
                total.accented_rate()
            ));
        }
    }
    assert!(failures.is_empty(), "OCR fell below its floor:\n{}", failures.join("\n"));
}
