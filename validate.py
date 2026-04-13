"""
validate.py — Compare an extracted Excel file against the ground truth CSVs.

Usage:
    python validate.py path/to/extracted_output.xlsx

Outputs:
    - Date coverage statistics (recall / precision)
    - Field-level similarity scores for text fields
    - A warning if the PDFs appear to cover a different period than the CSVs
"""

import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent

CSV_CONFIGS: dict[str, tuple[str, str]] = {
    "Teatro de la Cruz": str(
        SCRIPT_DIR
        / "Madrid Theater Database - Teatro de la Cruz and Teatro del Príncipe(Teatro de la Cruz) (3).csv"
    ),
    "Teatro del Príncipe": str(
        SCRIPT_DIR
        / "Madrid Theater Database - Teatro de la Cruz and Teatro del Príncipe(Teatro del Príncipe) (2).csv"
    ),
    "Unknown": str(
        SCRIPT_DIR
        / "Madrid Theater Database - Teatro de la Cruz and Teatro del Príncipe(Incomplete data (1750-1757)) (1).csv"
    ),
}

TEXT_FIELDS = ["Play 1 Title", "Author 1 Name", "Company Director"]
NUMERIC_FIELDS = ["Play 1 Box Office Receipts"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sim(a: str, b: str) -> float:
    """String similarity ratio 0–1."""
    return SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio()


def _load_ground_truth() -> pd.DataFrame:
    dfs = []
    for theater_name, csv_path in CSV_CONFIGS.items():
        p = Path(csv_path)
        if not p.exists():
            print(f"  [WARN] CSV not found: {csv_path}")
            continue
        df = pd.read_csv(p, encoding="latin-1")
        df = df.iloc[:, :16]   # drop trailing empty columns
        df.insert(0, "Theater", theater_name)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No ground truth CSVs found next to validate.py")
    return pd.concat(dfs, ignore_index=True)


# ── Main validation ───────────────────────────────────────────────────────────

def validate(extracted_path: str) -> None:
    print("\n" + "═" * 60)
    print("  Cartelera Teatral — Extraction Validator")
    print("═" * 60)

    print(f"\nLoading extracted data …  {extracted_path}")
    extracted = pd.read_excel(extracted_path)
    print(f"  {len(extracted):,} extracted records")

    print("\nLoading ground truth CSVs …")
    gt = _load_ground_truth()
    print(f"  {len(gt):,} ground truth records\n")

    # ── Date coverage ─────────────────────────────────────────────────────────
    gt_dates = set(gt["Performance Date"].dropna().str.strip())
    ext_dates = set(extracted["Performance Date"].dropna().str.strip())

    matched = gt_dates & ext_dates
    only_gt = gt_dates - ext_dates
    only_ext = ext_dates - gt_dates

    recall = len(matched) / len(gt_dates) * 100 if gt_dates else 0
    precision = len(matched) / len(ext_dates) * 100 if ext_dates else 0

    print("DATE COVERAGE")
    print(f"  Ground truth unique dates : {len(gt_dates):,}")
    print(f"  Extracted unique dates    : {len(ext_dates):,}")
    print(f"  Matching dates            : {len(matched):,}")
    print(f"  Date recall               : {recall:.1f}%   (how many GT dates were found)")
    print(f"  Date precision            : {precision:.1f}%   (how many extracted dates are valid)")
    if only_gt:
        sample = sorted(only_gt)[:5]
        print(f"  Dates in GT only (sample) : {sample}")
    if only_ext:
        sample = sorted(only_ext)[:5]
        print(f"  Dates in extract only     : {sample}")

    if not matched:
        print(
            "\n  ⚠  No matching dates found.\n"
            "  The PDF volumes may cover a different time period than the hand-labeled CSVs.\n"
            "  This is expected if the PDFs are volumes not yet in the CSV database.\n"
            "  Date coverage validation is not applicable — check a sample manually instead."
        )
        _print_sample(extracted)
        return

    # ── Field-level accuracy on matched dates ─────────────────────────────────
    print(f"\nFIELD ACCURACY  (on {len(matched):,} matching dates)")
    print(f"  {'Field':<32}  {'Avg similarity':>16}  {'Compared pairs':>16}")
    print("  " + "-" * 66)

    # Build lookup tables indexed by Performance Date
    # If multiple rows share the same date, take the first match
    gt_by_date = (
        gt[gt["Performance Date"].isin(matched)]
        .drop_duplicates(subset="Performance Date")
        .set_index("Performance Date")
    )
    ext_by_date = (
        extracted[extracted["Performance Date"].isin(matched)]
        .drop_duplicates(subset="Performance Date")
        .set_index("Performance Date")
    )

    for field in TEXT_FIELDS:
        if field not in gt.columns or field not in extracted.columns:
            print(f"  {field:<32}  {'(column missing)':>16}")
            continue

        scores: list[float] = []
        for date in matched:
            try:
                gt_val = str(gt_by_date.at[date, field]) if date in gt_by_date.index else ""
                ext_val = str(ext_by_date.at[date, field]) if date in ext_by_date.index else ""
                if gt_val.strip() and ext_val.strip() and gt_val != "nan" and ext_val != "nan":
                    scores.append(_sim(gt_val, ext_val))
            except Exception:
                continue

        if scores:
            avg = sum(scores) / len(scores)
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            print(f"  {field:<32}  {avg:>7.3f}  {bar}  ({len(scores):,} pairs)")
        else:
            print(f"  {field:<32}  {'(no comparable pairs)':>32}")

    # ── Numeric field accuracy ────────────────────────────────────────────────
    print(f"\nNUMERIC FIELD ACCURACY")
    for field in NUMERIC_FIELDS:
        if field not in gt.columns or field not in extracted.columns:
            continue
        exact = 0
        total = 0
        for date in matched:
            try:
                gt_val = str(gt_by_date.at[date, field]).strip() if date in gt_by_date.index else ""
                ext_val = str(ext_by_date.at[date, field]).strip() if date in ext_by_date.index else ""
                if gt_val and ext_val and gt_val != "nan" and ext_val != "nan":
                    total += 1
                    if gt_val == ext_val:
                        exact += 1
            except Exception:
                continue
        if total:
            print(f"  {field:<32}  exact match: {exact}/{total} = {exact/total*100:.1f}%")

    # ── Theater attribution ───────────────────────────────────────────────────
    if "Theater" in extracted.columns:
        print("\nTHEATER DISTRIBUTION (extracted)")
        for theater, count in extracted["Theater"].value_counts().items():
            print(f"  {theater:<30} {count:,} records")

    _print_sample(extracted)


def _print_sample(df: pd.DataFrame, n: int = 3) -> None:
    print(f"\nSAMPLE RECORDS (first {n})")
    sample_cols = [c for c in ["Theater", "Performance Date", "Play 1 Title",
                                "Author 1 Name", "Company Director"] if c in df.columns]
    print(df[sample_cols].head(n).to_string(index=False))
    print("\n" + "═" * 60 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  python validate.py  path/to/extracted_output.xlsx")
        sys.exit(1)
    validate(sys.argv[1])
