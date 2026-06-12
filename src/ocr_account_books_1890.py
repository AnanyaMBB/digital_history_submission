"""OCR pipeline for the Bank of England Daily Account Books, 1890.

Source. Bank of England Archive item C1/38, "DAILY ACCOUNTS FOR 'BOOKS' 1890",
1 volume of 79 leaves (digitised by the BoE Preservica system). The IIIF
manifest at
  https://boe.preservica.com/Render/render/resource/243c0067-06de-48ad-859e-b60c932a6c1a/universalviewer/manifest
yields 79 canvases. Canvas 1 is the leather cover, canvases 4 through about
75 are double-page weekly spreads of daily Balances (left) and Operations
(right), and the last 4 canvases are end-of-year summaries.

Each weekly spread covers 6 trading days (Wed through Tue, roughly) for one
calendar week. Layout. Left page = BALANCES (stock variables, Issue and
Banking Departments). Right page = OPERATIONS (daily flow variables,
including total Discounts, total Advances, Bank Rate). Numbers are
handwritten in ink with red-ink annotations for rate changes.

What this script does. It runs tesseract on each downloaded page image and
writes the raw OCR text to `outputs/ocr/raw_text/`. It also writes a JSON
metadata file recording which canvas index maps to which date range.

What this script does NOT do. It does NOT produce a clean daily CSV of the
balance-sheet values. The reason is that tesseract on these 1890 handwritten
ledger images is unreliable for cell-level numeric extraction. Numbers are
recorded in mixed handwriting styles, with crossed-out corrections and
red-ink annotations. A defensible cell-level extraction requires either a
multimodal vision-LLM pass or manual transcription. The hybrid extracted
daily CSV is produced in a separate step (see
`data/processed/boe_daily_1890.csv` and the manual-transcription notes in
`docs/ocr_plan_1890.md`).

The honest reading. tesseract here is useful for OCR-assisted page
navigation (date headers, row labels) and not for production cell-level
numeric extraction.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PAGES = ROOT / "data" / "raw" / "account_books_1890" / "pages"
OCR_OUT = ROOT / "outputs" / "ocr"
RAW_TEXT = OCR_OUT / "raw_text"
META = OCR_OUT / "metadata.json"

# Page-to-date mapping established by visual inspection of the high-res
# spreads. Each entry covers six trading days (Wed-Tue), the standard
# Bank of England weekly cycle for these ledgers.
PAGE_DATE_MAP: dict[str, dict] = {
    # canvas index, label, balances date range, operations date range
    "C1-38_063": {"canvas": 62, "balances": "1890-10-01 to 1890-10-07", "operations": "1890-10-02 to 1890-10-08"},
    "C1-38_064": {"canvas": 63, "balances": "1890-10-08 to 1890-10-14", "operations": "1890-10-09 to 1890-10-15"},
    "C1-38_065": {"canvas": 64, "balances": "1890-10-15 to 1890-10-21", "operations": "1890-10-16 to 1890-10-22"},
    "C1-38_066": {"canvas": 65, "balances": "1890-10-22 to 1890-10-28", "operations": "1890-10-23 to 1890-10-29"},
    "C1-38_067": {"canvas": 66, "balances": "1890-10-29 to 1890-11-04", "operations": "1890-10-30 to 1890-11-05"},
    "C1-38_068": {"canvas": 67, "balances": "1890-11-04 to 1890-11-11", "operations": "1890-11-05 to 1890-11-12"},
    "C1-38_069": {"canvas": 68, "balances": "1890-11-12 to 1890-11-18", "operations": "1890-11-13 to 1890-11-19",
                    "annotation": "Saturday Nov 15 1890 is the day the Lidderdale Guarantee Fund was finalised"},
    "C1-38_070": {"canvas": 69, "balances": "1890-11-19 to 1890-11-25", "operations": "1890-11-20 to 1890-11-26"},
    "C1-38_071": {"canvas": 70, "balances": "1890-11-26 to 1890-12-02", "operations": "1890-11-27 to 1890-12-03"},
    "C1-38_072": {"canvas": 71, "balances": "1890-12-03 to 1890-12-09", "operations": "1890-12-04 to 1890-12-10"},
    "C1-38_073": {"canvas": 72, "balances": "1890-12-10 to 1890-12-16", "operations": "1890-12-11 to 1890-12-17"},
    "C1-38_074": {"canvas": 73, "balances": "1890-12-17 to 1890-12-23", "operations": "1890-12-18 to 1890-12-24"},
    "C1-38_075": {"canvas": 74, "balances": "1890-12-24 to 1890-12-30", "operations": "1890-12-25 to 1890-12-31"},
    "C1-38_076": {"canvas": 75, "balances": "1890-12-31 (year-end)", "operations": "year-end summary entries"},
}


def preprocess(img_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Light preprocessing for old ink-on-paper ledgers.

    Returns two variants. A high-contrast grayscale image (good for printed
    headers) and a denoised inverted-binary image (better for handwritten
    numerals on yellowed paper).
    """
    bgr = cv2.imread(str(img_path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # High-contrast variant. CLAHE pulls out faint pencil marks too.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    # Adaptive threshold variant. Good for separating ink from paper.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=51, C=20,
    )
    return contrast, binary


def ocr_page(img_path: Path) -> dict:
    contrast, binary = preprocess(img_path)
    # Run tesseract on both variants and keep both outputs
    text_contrast = pytesseract.image_to_string(contrast, config="--psm 6")
    text_binary = pytesseract.image_to_string(binary, config="--psm 6")
    # Confidence per word from contrast variant
    data = pytesseract.image_to_data(
        contrast, config="--psm 6", output_type=pytesseract.Output.DICT
    )
    # In pytesseract >=0.3.10 confidences come back as ints already.
    raw_confs = data["conf"]
    confs = []
    for c in raw_confs:
        try:
            v = int(c)
            if v >= 0:
                confs.append(v)
        except (TypeError, ValueError):
            continue
    mean_conf = float(np.mean(confs)) if confs else float("nan")
    high_conf_words = [
        data["text"][i].strip()
        for i, c in enumerate(raw_confs)
        if isinstance(c, int) and c >= 70 and data["text"][i].strip()
    ]
    return {
        "text_contrast": text_contrast,
        "text_binary": text_binary,
        "mean_confidence": round(mean_conf, 1),
        "n_words_total": len(data["text"]),
        "n_words_high_confidence": len(high_conf_words),
        "high_confidence_words": high_conf_words[:50],
    }


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW_TEXT.mkdir(parents=True, exist_ok=True)
    OCR_OUT.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict] = {}
    pages = sorted(PAGES.glob("C1-38_*.jpg"))
    print(f"Processing {len(pages)} pages from {PAGES}")
    for p in pages:
        name = p.stem
        date_info = PAGE_DATE_MAP.get(name, {"canvas": None, "balances": "", "operations": ""})
        print(f"  {name} ...", end=" ", flush=True)
        try:
            result = ocr_page(p)
        except Exception as exc:
            print(f"FAILED: {exc}")
            metadata[name] = {**date_info, "ocr_status": f"failed: {exc}"}
            continue
        # Write raw text per page
        (RAW_TEXT / f"{name}.contrast.txt").write_text(result["text_contrast"])
        (RAW_TEXT / f"{name}.binary.txt").write_text(result["text_binary"])
        metadata[name] = {
            **date_info,
            "ocr_status": "ok",
            "image_sha256": sha256_of(p),
            "mean_confidence": result["mean_confidence"],
            "n_words_total": result["n_words_total"],
            "n_words_high_confidence": result["n_words_high_confidence"],
            "high_confidence_words_sample": result["high_confidence_words"][:20],
        }
        print(f"conf={result['mean_confidence']}  n_words={result['n_words_total']}")

    META.write_text(json.dumps(metadata, indent=2))
    print(f"\nWrote {META}")
    print(f"Wrote {len(pages)} raw text files to {RAW_TEXT}")

    # Summary of confidence statistics
    confs = [m["mean_confidence"] for m in metadata.values()
              if m.get("ocr_status") == "ok" and m.get("mean_confidence") == m.get("mean_confidence")]  # not nan
    if confs:
        print(f"\nMean confidence across pages: {np.mean(confs):.1f}")
        print(f"Pages above 50% confidence: {sum(1 for c in confs if c >= 50)} of {len(confs)}")


if __name__ == "__main__":
    main()
