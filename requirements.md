# Cartelera Teatral PDF → Excel Extractor
## Requirements Document

---

## 1. Project Overview

This tool automates the extraction of 18th-century theater performance records from
scanned PDF volumes of *Cartelera Teatral Madrileña* into a structured, consolidated
Excel file. It replaces a manual transcription workflow that previously produced three
separate spreadsheets — one per Madrid theater.

**Source data:**
- `Cartelera Teatral Madrileña 6 (1).pdf` (~123 MB scanned book)
- `Cartelera Teatral Madrileña 7 (2).pdf` (~138 MB scanned book)

**Ground truth / few-shot data (used for prompting and validation):**
- `Madrid Theater Database - ... (Teatro de la Cruz) (3).csv` — 2,576 rows
- `Madrid Theater Database - ... (Teatro del Príncipe) (2).csv` — 2,782 rows
- `Madrid Theater Database - ... (Incomplete data (1750-1757)) (1).csv` — 99 rows

---

## 2. Goals

| # | Goal |
|---|------|
| G1 | Extract ALL performance records from any given Cartelera Teatral PDF |
| G2 | Output a single Excel file with the same 16-column schema as the existing CSVs, plus a `Theater` column |
| G3 | Correctly attribute each record to `Teatro de la Cruz`, `Teatro del Príncipe`, or `Unknown` |
| G4 | Provide a GUI dialog so the user can select any PDF and choose the output path |
| G5 | Be resumable — save progress checkpoints so a crashed run can continue |
| G6 | Validate extraction quality against the hand-labeled CSVs |

---

## 3. Approach: LLM-Based Vision Extraction

### Why Claude Vision (not a fine-tuned model)

| Consideration | Claude Vision (chosen) | Fine-tuned Donut/LayoutLM |
|---|---|---|
| Historical Spanish fonts | Handled zero-shot | Requires labeled image↔JSON pairs |
| Variable / sparse fields | Handles naturally via context | Prone to hallucinating fixed slots |
| Alignment requirement | None — processes page as-is | Must map CSV rows → PDF page regions |
| Time to working system | Hours | Weeks |
| Cost at this dataset size | Acceptable (~$5–20/PDF) | High upfront engineering cost |
| When to revisit | After validating LLM accuracy | Phase 2 if cost becomes a blocker |

**Conclusion:** Use Claude claude-sonnet-4-6 (claude-sonnet-4-6) with vision. The 5,800
hand-labeled rows become few-shot examples and an evaluation set — not training data.
Fine-tuning is a Phase 2 option once the LLM baseline is validated.

### Theater Attribution Strategy

1. **Primary:** Ask Claude to detect theater section headers on each page  
   (e.g., "TEATRO DE LA CRUZ", running headers, chapter breaks)
2. **Stateful carry-forward:** Once a theater is identified, all subsequent pages
   inherit it until a new header overrides it
3. **Fallback:** If no header is detected, carry forward the last known theater value

---

## 4. Output Schema

The output Excel file has **17 columns** — the 16 from the existing CSVs plus `Theater`
as the first column.

| Column | Type | Notes |
|--------|------|-------|
| **Theater** | string | `Teatro de la Cruz` \| `Teatro del Príncipe` \| `Unknown` |
| Performance Date | string | As printed in Spanish, e.g. `14 Abril 1748` |
| Play 1 Title | string | Main comedia/play title |
| Author 1 Name | string | Author of play 1 |
| Play 1 Genre | string | comedia, zarzuela, sainete, auto sacramental, etc. |
| Play 1 # of Acts | string | Integer as string, e.g. `3` |
| Play 1 Date of Premier | string | Year or date of first known performance |
| Play 1 Box Office Receipts | string | Numeric, in reales |
| Play 2 Title | string | Second piece title (often sainete or entremés) |
| Author 2 Name | string | Author of play 2 |
| Play 2 Genre | string | Genre of play 2 |
| Play 2 # of Acts | string | Number of acts for play 2 |
| Play 2 Date of Premier | string | Premiere date of play 2 |
| Play 2 Box Office Receipts | string | Box office receipts for play 2 |
| Play 3 | string | Third piece title (loa, fin de fiesta, etc.) |
| Company Director | string | Name of the acting company director |
| Day of the Week | string | Day name in English (Monday–Sunday) |

Empty fields are stored as empty strings `""`.

---

## 5. Few-Shot Prompting Strategy

The 3 ground truth CSVs are used to give Claude concrete examples of expected
output format and content patterns:

- **N = 4 rows sampled per CSV** → 12 total examples in the system prompt
- Sampling strategy: prefer rows with the most non-null fields (richest examples)
- Examples are serialized as a JSON array matching the extraction output schema
- The few-shot block appears once in the user message (not per-page, to save tokens)
- Examples are statically selected at startup and reused across all pages

---

## 6. Processing Pipeline

```
PDF file
  │
  ▼
[1] Render each page to PNG at 150 DPI using PyMuPDF
  │
  ▼
[2] For each page (in order):
    │
    ├─ Build user message:
    │    - Few-shot examples (JSON block)
    │    - Current theater context (carry-forward state)
    │    - Page image (base64 PNG)
    │
    ├─ POST to Claude claude-sonnet-4-6 (claude-sonnet-4-6) Messages API
    │
    ├─ Parse JSON response:
    │    { "theater_detected": "...", "records": [...] }
    │
    ├─ Update theater state if header detected
    │
    ├─ Append records (filling theater from state if record is missing it)
    │
    └─ Save checkpoint every 10 pages
  │
  ▼
[3] Build pandas DataFrame with OUTPUT_COLUMNS schema
  │
  ▼
[4] Write to .xlsx with formatted headers, auto-width columns, frozen top row
  │
  ▼
[5] Delete checkpoint file on success
```

---

## 7. Functional Requirements

### FR-1: PDF Rendering
- Convert each PDF page to a PNG image at 150 DPI
- Use PyMuPDF (`fitz`) for rendering — no external system dependencies

### FR-2: Claude Vision Extraction
- Model: `claude-sonnet-4-6` (best vision + instruction following)
- Max tokens per response: 4096
- Retry on rate limits and 5xx errors with exponential backoff (max 3 attempts)
- On JSON parse failure: retry with clarification, fall back to empty record set
- Per-page timeout: handled by SDK defaults

### FR-3: Theater Attribution
- Detect theater name from page section headers
- Carry forward theater state across pages
- Apply state as default when Claude returns `theater = "Unknown"` or null

### FR-4: Checkpoint / Resume
- Save progress to `<output_name>.checkpoint.json` every 10 pages
- Format: `{ "records": [...], "page_num": N, "current_theater": "..." }`
- On startup, detect existing checkpoint and offer to resume from last saved page
- Delete checkpoint file upon successful completion

### FR-5: Output Excel
- Write using `openpyxl`
- Apply header formatting: bold white text on dark blue (#2E4057) background
- Auto-size column widths (capped at 40 chars)
- Freeze top row (`A2`)
- All 17 output columns always present, even if empty

### FR-6: Validation (validate.py)
- Load extracted Excel + all 3 ground truth CSVs
- Match records by `Performance Date` string
- Report: date recall %, field similarity scores for Play 1 Title, Author 1 Name,
  Company Director
- Warn if no dates overlap (PDFs may cover different periods than CSVs)

### FR-7: GUI Application
- Built with `tkinter` (stdlib — no extra dependency)
- Fields: API key input (masked), PDF path, output path
- Pre-fill API key from `ANTHROPIC_API_KEY` environment variable if set
- Progress bar (0–100%) updated per page
- Status label showing current page / total pages
- "Extract to Excel" button (disabled while running)
- Success dialog showing record count and output path
- Error dialog with message + pointer to `extractor.log`
- Log file (`extractor.log`) written alongside the script

---

## 8. Non-Functional Requirements

| # | Requirement |
|---|-------------|
| NFR-1 | All Python 3.10+ — no system dependencies beyond pip packages |
| NFR-2 | Works on macOS, Windows, Linux (tkinter is cross-platform stdlib) |
| NFR-3 | API key never written to disk, only held in memory during run |
| NFR-4 | Gracefully skips pages that fail extraction (logs error, continues) |
| NFR-5 | Checkpoint prevents full re-run if process is interrupted |
| NFR-6 | All log output goes to `extractor.log` at INFO level |

---

## 9. Cost & Performance Estimates

| PDF size estimate | Pages | API calls | Approx. cost (claude-sonnet-4-6) | Approx. runtime |
|---|---|---|---|---|
| Small (50 pages) | 50 | 50 | ~$1–3 | ~5 min |
| Medium (200 pages) | 200 | 200 | ~$5–10 | ~20 min |
| Large (500 pages) | 500 | 500 | ~$15–25 | ~50 min |

*Based on ~1,500 input tokens (image) + 500 output tokens per page at claude-sonnet-4-6 pricing.*

---

## 10. File Structure

```
playlistingtranscriber/
├── requirements.md                  ← this document
├── requirements.txt                 ← pip dependencies
├── extractor.py                     ← core pipeline (PDF → records via Claude)
├── main.py                          ← GUI entry point
├── validate.py                      ← validation against ground truth CSVs
├── extractor.log                    ← generated at runtime
│
├── Cartelera Teatral Madrileña 6 (1).pdf
├── Cartelera Teatral Madrileña 7 (2).pdf
│
├── Madrid Theater Database - ...(Teatro de la Cruz) (3).csv
├── Madrid Theater Database - ...(Teatro del Príncipe) (2).csv
└── Madrid Theater Database - ...(Incomplete data (1750-1757)) (1).csv
```

---

## 11. Dependencies

```
anthropic>=0.36.0        # Claude API SDK
PyMuPDF>=1.24.0          # PDF rendering (no system deps required)
Pillow>=10.3.0           # Image handling
pandas>=2.2.0            # DataFrame operations
openpyxl>=3.1.2          # Excel read/write and formatting
```

`tkinter` is part of Python's standard library (included with Python 3.x on all platforms).

---

## 12. Running Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Run the GUI app
python main.py

# Validate extracted output against ground truth
python validate.py path/to/output.xlsx
```

Or set the API key in your environment to avoid pasting it each time:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python main.py
```

---

## 13. Phase 2 (Future)

If API cost becomes a concern at large scale, consider fine-tuning:

1. Use LLM-extracted + human-verified records as pseudo-labels
2. Fine-tune **Donut** (Document Understanding Transformer) on aligned page → JSON pairs
3. Donut is OCR-free and end-to-end — suitable for consistent historical print layouts
4. Target: ~500 labeled page images to achieve comparable accuracy at zero inference cost

This is only worth pursuing after the LLM baseline is validated and a consistent failure
mode is identified.
