"""
extractor.py — Core pipeline for Cartelera Teatral PDF extraction.

Renders PDF pages to images, sends them to Claude Vision, and returns
a pandas DataFrame of structured theater performance records.
"""

import base64
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, Optional

import fitz  # PyMuPDF
import pandas as pd
from google import genai
from google.genai import types as genai_types
from PIL import Image

logger = logging.getLogger(__name__)

# ── Output schema ─────────────────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    "Theater",
    "Performance Date",
    "Play 1 Title",
    "Author 1 Name",
    "Play 1 Genre",
    "Play 1 # of Acts",
    "Play 1 Date of Premier",
    "Play 1 Box Office Receipts",
    "Play 2 Title",
    "Author 2 Name",
    "Play 2 Genre",
    "Play 2 # of Acts",
    "Play 2 Date of Premier",
    "Play 2 Box Office Receipts",
    "Play 3",
    "Company Director",
    "Day of the Week",
]

# Claude JSON field names → Excel column names
FIELD_TO_COLUMN = {
    "theater":            "Theater",
    "performance_date":   "Performance Date",
    "play_1_title":       "Play 1 Title",
    "author_1_name":      "Author 1 Name",
    "play_1_genre":       "Play 1 Genre",
    "play_1_acts":        "Play 1 # of Acts",
    "play_1_premiere":    "Play 1 Date of Premier",
    "play_1_receipts":    "Play 1 Box Office Receipts",
    "play_2_title":       "Play 2 Title",
    "author_2_name":      "Author 2 Name",
    "play_2_genre":       "Play 2 Genre",
    "play_2_acts":        "Play 2 # of Acts",
    "play_2_premiere":    "Play 2 Date of Premier",
    "play_2_receipts":    "Play 2 Box Office Receipts",
    "play_3":             "Play 3",
    "company_director":   "Company Director",
    "day_of_week":        "Day of the Week",
}

# ── Few-shot loading ──────────────────────────────────────────────────────────

def load_few_shot_examples(
    csv_configs: dict[str, tuple[str, str]],
    n_per_file: int = 2,
) -> list[dict]:
    """
    Load representative examples from the ground truth CSVs.

    Picks the n_per_file rows with the most non-null fields from each CSV
    so Claude sees well-populated examples. Also includes one sparse row
    so Claude knows to leave fields empty rather than hallucinate.

    Args:
        csv_configs: { key: (csv_path, theater_name) }
        n_per_file:  number of rows to sample per CSV

    Returns:
        List of dicts in the JSON extraction schema.
    """
    col_to_field = {v: k for k, v in FIELD_TO_COLUMN.items()}
    examples: list[dict] = []

    for _key, (csv_path, theater_name) in csv_configs.items():
        p = Path(csv_path)
        if not p.exists():
            logger.warning("CSV not found, skipping: %s", csv_path)
            continue

        df = pd.read_csv(p, encoding="latin-1")
        # Keep only the 16 named schema columns (drop trailing empties)
        df = df[[c for c in OUTPUT_COLUMNS[1:] if c in df.columns]]
        df = df.dropna(how="all")

        # Select the richest rows first
        completeness = df.notna().sum(axis=1)
        rich_idx = completeness.nlargest(min(n_per_file, len(df))).index
        sample = df.loc[rich_idx]

        for _, row in sample.iterrows():
            record: dict = {"theater": theater_name}
            for col_name, field_name in col_to_field.items():
                if col_name == "Theater":
                    continue
                val = row.get(col_name, "")
                record[field_name] = "" if pd.isna(val) else str(val).strip()
            examples.append(record)

    return examples


def _format_few_shot_block(examples: list[dict]) -> str:
    """Serialise examples as a JSON block for inclusion in the user message."""
    if not examples:
        return ""
    return (
        "\n\nEXAMPLE RECORDS from this database — use these as the reference "
        "for expected output format, field names, and content patterns:\n"
        + json.dumps(examples, ensure_ascii=False, indent=2)
        + "\n"
    )


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert at reading 18th-century Spanish theater records. You are \
processing scanned pages from "Cartelera Teatral Madrileña", a multi-volume \
printed reference work cataloguing performances at the two main Madrid \
municipal theaters (Teatro de la Cruz and Teatro del Príncipe), roughly 1748–1800.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW THESE PAGES ARE ORGANISED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each page has one or more SECTIONS. Every section looks like this:

  TEATRO DE LA CRUZ  [or TEATRO DEL PRÍNCIPE]   [YEAR or SEASON, e.g. 1762-1763]
  [Company Director name]

  [Play title]  ([Author])  [genre / description]
    [Day₁]  [Day₂]  [Day₃]  …       ← day-of-month numbers for each performance
    [Rec₁]  [Rec₂]  [Rec₃]  …       ← box-office receipts (in reales) per day

  [Next play title]  …
    [Day₁]  [Day₂]  …
    [Rec₁]  [Rec₂]  …

  [sainete / entremés / secondary piece title]
    (same day numbers as the main play above — performed the same evenings)

The month is shown in column-headers, or can be inferred from surrounding \
date context. The year is in the section header.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK — ONE OUTPUT RECORD PER PERFORMANCE DATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: A play that ran on 10 different days must produce 10 separate \
records — one for each day. Match day₁ → receipt₁, day₂ → receipt₂, etc.

Each record represents ONE EVENING at the theater and has:
  • play_1_*  fields  → the main play (comedia, zarzuela, auto sacramental …)
  • play_2_*  fields  → the secondary piece performed the SAME evening \
                        (sainete, entremés, loa …)
  • play_3           → a third piece title if present (fin de fiesta, etc.)

When a sainete/entremés is listed for the same dates as a main comedia, \
combine them into the play_1 / play_2 slots of each shared date's record.

If a block only lists a secondary piece with no main play visible, put the \
title in play_1_title (do not leave play_1_title empty and put it in play_2).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  theater            – "Teatro de la Cruz", "Teatro del Príncipe", or "Unknown"
  performance_date   – reconstructed full date: "DD Mes YYYY", e.g. "14 Abril 1748"
  play_1_title       – title of the main comedia / play (keep leading * for premieres)
  author_1_name      – author of play 1
  play_1_genre       – genre (comedia, zarzuela, sainete, auto sacramental, …)
  play_1_acts        – number of acts as a string, e.g. "3"
  play_1_premiere    – year or date of first known performance
  play_1_receipts    – box office receipts for that SPECIFIC evening (numeric string)
  play_2_title       – secondary piece title for the same evening
  author_2_name      – author of play 2
  play_2_genre       – genre of play 2 (usually sainete, entremés, loa …)
  play_2_acts        – number of acts for play 2
  play_2_premiere    – premiere date of play 2
  play_2_receipts    – receipts for play 2 on that specific evening
  play_3             – third piece title if present
  company_director   – director of the acting company (from section header)
  day_of_week        – day of the week IN ENGLISH calculated from the full date \
                       (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Produce ONE record per performance date — never merge multiple dates into one.
2. Use "" for any field that is blank, illegible, or genuinely not present.
3. Report the theater name found in section headers via "theater_detected".
4. Spanish months: Enero=Jan, Febrero=Feb, Marzo=Mar, Abril=Apr, Mayo=May, \
   Junio=Jun, Julio=Jul, Agosto=Aug, Septiembre=Sep, Octubre=Oct, \
   Noviembre=Nov, Diciembre=Dec. Keep performance_date in Spanish.
5. Asterisks (*) before titles indicate a premiere night — preserve them.
6. Do NOT invent values. Blank is correct when data is absent or illegible.
7. Return ONLY valid JSON — no prose, no markdown fences.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXACT RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "theater_detected": "<theater from section header, or null>",
  "season_year": "<year or range from section header, e.g. 1762-1763, or null>",
  "records": [
    {
      "theater": "…",
      "performance_date": "…",
      "play_1_title": "…",
      "author_1_name": "…",
      "play_1_genre": "…",
      "play_1_acts": "…",
      "play_1_premiere": "…",
      "play_1_receipts": "…",
      "play_2_title": "…",
      "author_2_name": "…",
      "play_2_genre": "…",
      "play_2_acts": "…",
      "play_2_premiere": "…",
      "play_2_receipts": "…",
      "play_3": "…",
      "company_director": "…",
      "day_of_week": "…"
    }
  ]
}
"""


# ── PDF utilities ─────────────────────────────────────────────────────────────

def pdf_to_images(
    pdf_path: str,
    dpi: int = 220,
    auto_rotate: bool = True,
    split_spreads: bool = True,
) -> list[Image.Image]:
    """
    Render every page of a PDF as PIL RGB Images.

    This PDF's scans are sideways two-page spreads: each PDF page contains two
    physical book pages rotated 90° CCW. We rotate upright and split into halves
    so each returned image is one physical book page, legible by Claude.

    Args:
        dpi:           Render DPI. 220 keeps per-half JPEGs well under 5 MB.
        auto_rotate:   Rotate portrait-oriented pages 90° CCW so text is upright.
        split_spreads: After rotation, cut each landscape page into left+right halves.

    Returns the list of logical pages in reading order (left half then right half).
    """
    doc = fitz.open(pdf_path)
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    images: list[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if auto_rotate and img.height > img.width:
            img = img.rotate(90, expand=True)

        if split_spreads and img.width > img.height:
            mid = img.width // 2
            images.append(img.crop((0, 0, mid, img.height)))
            images.append(img.crop((mid, 0, img.width, img.height)))
        else:
            images.append(img)
    doc.close()
    logger.info(
        "Rendered %d logical pages from %s at %d DPI (auto_rotate=%s split=%s)",
        len(images), pdf_path, dpi, auto_rotate, split_spreads,
    )
    return images


# Claude API hard limit is 5 MB base64-encoded. Target a safe ceiling.
_B64_BUDGET_BYTES = 4_500_000


def _image_to_base64(img: Image.Image) -> tuple[str, str]:
    """
    Encode a PIL image as JPEG at the highest quality that fits Claude's
    5 MB base64 budget. Only downscales as a last resort — we want Claude
    to see the text at native resolution.

    Returns (base64_string, media_type).
    """
    def encode(im: Image.Image, q: int) -> bytes:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q, optimize=True)
        return buf.getvalue()

    # 1) Try quality steps at native resolution.
    for q in (92, 85, 78, 70, 60):
        raw = encode(img, q)
        b64 = base64.standard_b64encode(raw)
        if len(b64) <= _B64_BUDGET_BYTES:
            logger.debug(
                "Encoded %dx%d JPEG q=%d → %.0f KB raw / %.0f KB b64",
                img.width, img.height, q, len(raw) / 1024, len(b64) / 1024,
            )
            return b64.decode("utf-8"), "image/jpeg"

    # 2) Still too big: downscale iteratively.
    scaled = img
    for _ in range(6):
        scaled = scaled.resize(
            (int(scaled.width * 0.85), int(scaled.height * 0.85)), Image.LANCZOS
        )
        raw = encode(scaled, 75)
        b64 = base64.standard_b64encode(raw)
        if len(b64) <= _B64_BUDGET_BYTES:
            logger.warning(
                "Downscaled to %dx%d to fit budget (%.0f KB b64)",
                scaled.width, scaled.height, len(b64) / 1024,
            )
            return b64.decode("utf-8"), "image/jpeg"

    raise RuntimeError("Unable to fit image under 5 MB base64 budget")


def _image_to_jpeg_bytes(img: Image.Image, target_bytes: int = 4_500_000) -> bytes:
    """Encode a PIL image as JPEG bytes under `target_bytes` (Gemini inline limit)."""
    def encode(im: Image.Image, q: int) -> bytes:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q, optimize=True)
        return buf.getvalue()

    for q in (92, 85, 78, 70, 60):
        raw = encode(img, q)
        if len(raw) <= target_bytes:
            logger.debug("Encoded %dx%d JPEG q=%d → %.0f KB",
                         img.width, img.height, q, len(raw) / 1024)
            return raw

    scaled = img
    for _ in range(6):
        scaled = scaled.resize(
            (int(scaled.width * 0.85), int(scaled.height * 0.85)), Image.LANCZOS
        )
        raw = encode(scaled, 75)
        if len(raw) <= target_bytes:
            logger.warning("Downscaled to %dx%d to fit Gemini budget (%.0f KB)",
                           scaled.width, scaled.height, len(raw) / 1024)
            return raw
    raise RuntimeError("Unable to fit image under Gemini inline image budget")


# ── Response schema for Gemini structured output ──────────────────────────────

_RECORD_FIELDS = [
    "theater", "performance_date",
    "play_1_title", "author_1_name", "play_1_genre", "play_1_acts",
    "play_1_premiere", "play_1_receipts",
    "play_2_title", "author_2_name", "play_2_genre", "play_2_acts",
    "play_2_premiere", "play_2_receipts",
    "play_3", "company_director", "day_of_week",
]

RECORD_SCHEMA = {
    "type": "OBJECT",
    "properties": {k: {"type": "STRING"} for k in _RECORD_FIELDS},
    "required": ["theater", "performance_date"],
    "property_ordering": _RECORD_FIELDS,
}

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "theater_detected": {"type": "STRING", "nullable": True},
        "season_year": {"type": "STRING", "nullable": True},
        "director_detected": {"type": "STRING", "nullable": True},
        "section_continues_from_previous": {"type": "BOOLEAN"},
        "records": {"type": "ARRAY", "items": RECORD_SCHEMA},
    },
    "required": [
        "theater_detected", "season_year", "director_detected",
        "section_continues_from_previous", "records",
    ],
    "property_ordering": [
        "theater_detected", "season_year", "director_detected",
        "section_continues_from_previous", "records",
    ],
}


# ── Claude API call ────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.5-flash"


def _canonical_theater(name: str) -> str:
    """Normalize any casing/accent variant to the canonical theater name."""
    if not name:
        return ""
    n = name.strip().lower().replace("í", "i")
    if "cruz" in n:
        return "Teatro de la Cruz"
    if "principe" in n or "príncipe" in name.lower():
        return "Teatro del Príncipe"
    return name.strip()


def _extract_page_records(
    client: genai.Client,
    image: Image.Image,
    few_shot_block: str,
    page_index: int,
    total_pages: int,
    current_theater: str,
    current_director: str,
    current_season: str,
    max_retries: int = 6,
) -> dict:
    """
    Send one page image to Gemini 2.5 Flash with response_schema-constrained
    JSON output and return the parsed dict.
    """
    user_text = (
        f"Page {page_index + 1} of {total_pages}.\n"
        f"CONTEXT FROM PREVIOUS PAGE (carry forward if this page has no new section header):\n"
        f"  • theater  = {current_theater}\n"
        f"  • director = {current_director}\n"
        f"  • season   = {current_season}\n"
        f"Set section_continues_from_previous=true if this page lacks a fresh section "
        f"header and the records belong to the carried-forward section. In that case, "
        f"populate each record's theater/company_director/season from the context above.\n"
        f"If this page has its OWN section header, extract theater_detected, "
        f"season_year, and director_detected from that header and set "
        f"section_continues_from_previous=false.\n"
        f"Remember: produce ONE record per performance DATE — expand multi-date play blocks.\n"
        f"{few_shot_block}"
        "Extract all performance records from this page as structured JSON."
    )

    raw_jpeg = _image_to_jpeg_bytes(image)
    contents = [
        genai_types.Part.from_bytes(data=raw_jpeg, mime_type="image/jpeg"),
        user_text,
    ]

    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        max_output_tokens=32768,
        temperature=0.0,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )

            usage = getattr(response, "usage_metadata", None)
            if usage:
                logger.info(
                    "Page %d usage: in=%s out=%s",
                    page_index + 1,
                    getattr(usage, "prompt_token_count", "?"),
                    getattr(usage, "candidates_token_count", "?"),
                )

            text = response.text
            if not text:
                raise ValueError("Empty response text from Gemini")

            result = json.loads(text)

            logger.info(
                "Page %d: OK — %d records, theater=%s, season=%s, director=%s, continues=%s",
                page_index + 1,
                len(result.get("records", [])),
                result.get("theater_detected"),
                result.get("season_year"),
                result.get("director_detected"),
                result.get("section_continues_from_previous"),
            )
            if not result.get("records"):
                logger.warning("Page %d: empty records array", page_index + 1)

            result.setdefault("records", [])
            result.setdefault("theater_detected", None)
            result.setdefault("season_year", None)
            result.setdefault("director_detected", None)
            result.setdefault("section_continues_from_previous", False)
            return result

        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error on attempt %d: %s", attempt + 1, exc)
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(2.0)

        except Exception as exc:
            msg = str(exc)
            retriable = any(s in msg for s in (
                "429", "500", "502", "503", "504",
                "RESOURCE_EXHAUSTED", "UNAVAILABLE",
                "Connection reset", "ConnectionError",
            ))
            if retriable and attempt < max_retries - 1:
                wait = min(180.0, 5.0 * (3 ** attempt))
                logger.warning("Transient error on attempt %d (%s) — sleeping %.0fs",
                               attempt + 1, exc.__class__.__name__, wait)
                time.sleep(wait)
                last_err = exc
            else:
                raise

    raise RuntimeError(
        f"Page {page_index + 1} failed after {max_retries} attempts. Last error: {last_err}"
    )


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    records: list[dict],
    completed_page: int,
    current_theater: str,
    current_director: str,
    current_season: str,
    path: str,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "records": records,
                "completed_page": completed_page,
                "current_theater": current_theater,
                "current_director": current_director,
                "current_season": current_season,
            },
            f,
            ensure_ascii=False,
        )
    logger.debug("Checkpoint saved at page %d", completed_page)


def load_checkpoint(path: str) -> tuple[list[dict], int, str, str, str]:
    """
    Returns (records, last_completed_page_index, current_theater,
             current_director, current_season).
    Returns ([], 0, "Unknown", "", "") if no checkpoint file exists.
    """
    p = Path(path)
    if not p.exists():
        return [], 0, "Unknown", "", ""

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    page = data.get("completed_page", 0)
    theater = data.get("current_theater", "Unknown")
    director = data.get("current_director", "")
    season = data.get("current_season", "")
    logger.info(
        "Resuming from checkpoint: page %d, theater=%s, director=%s, records so far=%d",
        page, theater, director, len(records),
    )
    return records, page, theater, director, season


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: str,
    api_key: str,
    few_shot_examples: list[dict],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    checkpoint_path: Optional[str] = None,
    pages_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Process a PDF and return a DataFrame of performance records.

    Args:
        pdf_path:           Path to the input PDF.
        api_key:            Google Gemini API key.
        few_shot_examples:  Output of load_few_shot_examples().
        progress_callback:  Called as (current_page, total_pages, status_msg).
        checkpoint_path:    JSON file for saving/resuming progress.
        pages_limit:        If set, process only the first N pages (for evaluation).

    Returns:
        DataFrame with OUTPUT_COLUMNS schema.
    """
    client = genai.Client(api_key=api_key)
    few_shot_block = _format_few_shot_block(few_shot_examples)

    # Resume from checkpoint if available
    all_records: list[dict] = []
    start_page = 0
    current_theater = "Unknown"
    current_director = ""
    current_season = ""

    if checkpoint_path:
        (all_records, start_page, current_theater,
         current_director, current_season) = load_checkpoint(checkpoint_path)

    # Render pages
    if progress_callback:
        progress_callback(0, 1, "Rendering PDF pages to images…")

    images = pdf_to_images(pdf_path)
    total_in_pdf = len(images)

    # Apply pages_limit AFTER rendering so we know the real total
    if pages_limit is not None:
        total = min(pages_limit, total_in_pdf)
        logger.info("pages_limit=%d applied — processing %d of %d pages",
                    pages_limit, total, total_in_pdf)
    else:
        total = total_in_pdf

    if start_page >= total:
        logger.info("All %d pages already processed (checkpoint).", total)
    else:
        logger.info("Processing pages %d–%d of %d", start_page + 1, total, total_in_pdf)

    pages_attempted = 0
    last_success_idx = start_page - 1
    # Each record is tagged with a section id; we increment on a fresh section
    # header so backfill never crosses a boundary.
    section_id = 0
    for page_idx in range(start_page, total):
        pages_attempted += 1
        msg = f"Extracting page {page_idx + 1} of {total}…"
        if progress_callback:
            progress_callback(page_idx + 1, total, msg)
        logger.info(msg)

        try:
            result = _extract_page_records(
                client=client,
                image=images[page_idx],
                few_shot_block=few_shot_block,
                page_index=page_idx,
                total_pages=total,
                current_theater=current_theater,
                current_director=current_director,
                current_season=current_season,
            )

            # Fresh section headers bump the section_id so backfill is scoped.
            continues = result.get("section_continues_from_previous", False)

            detected_theater = _canonical_theater(result.get("theater_detected") or "")
            new_theater = detected_theater and detected_theater != current_theater

            detected_director = (result.get("director_detected") or "").strip()
            new_director = detected_director and detected_director != current_director

            if (new_theater or new_director) and not continues:
                section_id += 1
                logger.info("New section #%d at page %d", section_id, page_idx + 1)

            if detected_theater:
                current_theater = detected_theater
                logger.info("Theater → %s (page %d)", current_theater, page_idx + 1)

            if detected_director:
                current_director = detected_director
                logger.info("Director → %s (page %d)", current_director, page_idx + 1)

            detected_season = result.get("season_year")
            if detected_season and detected_season not in ("null", None, ""):
                current_season = detected_season
                logger.info("Season → %s (page %d)", current_season, page_idx + 1)

            # Collect records; carry forward canonical theater + director.
            for rec in result.get("records", []):
                rec["theater"] = _canonical_theater(rec.get("theater") or "") or current_theater
                if not rec.get("company_director") or rec["company_director"] in ("", "null"):
                    rec["company_director"] = current_director
                rec["_section_id"] = section_id
                all_records.append(rec)

            logger.debug("Page %d: extracted %d records", page_idx + 1,
                         len(result.get("records", [])))
            last_success_idx = page_idx

        except Exception as exc:
            logger.error("Skipping page %d due to error: %s", page_idx + 1, exc)
            # Continue — one bad page should not abort the whole run.
            # Do NOT advance last_success_idx so a rerun will retry this page.

        # Checkpoint every 10 pages — only through the last contiguously
        # successful page, so failed pages get retried on resume.
        if checkpoint_path and (page_idx + 1) % 10 == 0 and last_success_idx >= 0:
            save_checkpoint(all_records, last_success_idx + 1, current_theater,
                            current_director, current_season, checkpoint_path)

    # Final checkpoint — persist progress up to the last successful page only.
    if checkpoint_path and last_success_idx >= start_page:
        save_checkpoint(all_records, last_success_idx + 1, current_theater,
                        current_director, current_season, checkpoint_path)
    elif checkpoint_path and pages_attempted > 0 and last_success_idx < start_page:
        logger.warning(
            "Not writing checkpoint: %d pages attempted, 0 succeeded. "
            "Fix the error and rerun.", pages_attempted,
        )

    # Retroactive backfill — scoped to each section so values never leak across
    # a Teatro / director boundary. The very first section (section_id==0) is
    # the special case: if the PDF opens mid-section, its theater/director
    # aren't known until a later page reveals them, so we also propagate
    # forward-learned values backward within that opening section.
    def _backfill(field: str, empty_vals: tuple[str, ...]) -> None:
        by_section: dict[int, str] = {}
        for rec in all_records:
            sid = rec.get("_section_id", 0)
            val = rec.get(field, "")
            if val and val not in empty_vals and sid not in by_section:
                by_section[sid] = val
        for rec in all_records:
            if not rec.get(field) or rec.get(field) in empty_vals:
                sid = rec.get("_section_id", 0)
                if sid in by_section:
                    rec[field] = by_section[sid]

    _backfill("theater", ("", "Unknown", "null"))
    _backfill("company_director", ("", "null"))

    # Drop internal bookkeeping before DataFrame construction.
    for rec in all_records:
        rec.pop("_section_id", None)

    # Build DataFrame. All fields are strings — force object dtype so pandas
    # doesn't coerce receipt strings like "1449" into floats (which round-trip
    # through Excel as "1449.0" and break field-level comparisons).
    df = pd.DataFrame(all_records, dtype=object)
    if not df.empty:
        df = df.rename(columns=FIELD_TO_COLUMN)

    # Ensure every expected column exists (fills missing with empty string)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[OUTPUT_COLUMNS].reset_index(drop=True)
    # Replace any stray NaN / None / float-coerced values with clean strings.
    df = df.fillna("").astype(str).replace({"nan": "", "None": ""})
    # Strip a trailing ".0" that sneaks in when pandas coerces numeric-looking
    # strings to floats before we override the dtype.
    for col in OUTPUT_COLUMNS:
        df[col] = df[col].str.replace(r"\.0$", "", regex=True)
    logger.info("Extraction complete: %d total records from %d pages", len(df), total)
    return df
