# -*- coding: utf-8 -*-
"""
Build two CRISPRi tables with robust parsing for a **single-sheet** map that
contains **multiple plates (LIB#)** laid out as blocks, each with a 96‑well grid.

It also works for multi-sheet workbooks, but the primary target is one sheet
that embeds many plates.

Inputs (same folder as this script unless you change paths):
- CRISPRi library map.xlsx
- scoelicolor_annotation_with_regulatory_links_v2.csv

Outputs:
- CRISPRi_library_targets_with_annotations.csv
- annotation_plus_CRISPRi_positions.csv
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# =========================
# --- Config / Paths ------
# =========================
LIB_EXCEL_PATH = Path("CRISPRi library map.xlsx")
ANNO_CSV_PATH = Path("scoelicolor_annotation_with_regulatory_links_v2.csv")

# If auto detection is wrong for some blocks/sheets, override here:
# key: sheet name OR synthetic key "sheet@r,c" (anchor cell position), value: int plate number
MANUAL_LIB_MAP: dict[str, int] = {
    # "Sheet1": 3,
    # "Sheet1@12,2": 7,  # anchor at (row=12, col=2) should be LIB#7
}

# Fallback numbering by encounter order when no LIB number can be found
FALLBACK_LIB_BY_ORDER = True

# If a block is not a 96-well grid but a list-like table, use these regexes
FALLBACK_WELL_COL_REGEX = r"(well|pos|position)"
FALLBACK_GENE_COL_REGEX = r"(sco|gene|target|locus)"

# Search windows around an anchor (in rows & cols)
ROW_WINDOW_PRIMARY = 80   # rows below the anchor to search for a grid
COL_WINDOW_PRIMARY = 40   # cols to the right of the anchor
ROW_WINDOW_EXPANDED = 160  # expanded retry rows
COL_WINDOW_EXPANDED = 80   # expanded retry cols

# =========================
# --- Utilities -----------
# =========================


def is_empty(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    s = str(x).strip()
    if s == "":
        return True
    if s.lower() in {"empty", "ntc", "control", "blank", "na", "n/a", "none"}:
        return True
    if re.fullmatch(r"[-–—]+", s):
        return True
    return False


def extract_sco_ids(cell) -> List[str]:
    """Extract all SCO-like ids in a cell (case-insensitive, allows 'SCO-1234', 'sco 1234')."""
    if cell is None:
        return []
    txt = str(cell)
    nums = re.findall(r"(?:SCO|sco|Sco)\s*[-_]*\s*(\d{3,5})", txt)
    return [f"SCO{n.zfill(4)}" for n in nums]


def normalize_to_sco_id(x) -> Optional[str]:
    if pd.isna(x):
        return None
    m = re.search(r"\bSCO\s*[-_]*\s*(\d{3,5})\b", str(x), flags=re.I)
    return f"SCO{m.group(1).zfill(4)}" if m else None


PATTERN_LIB = re.compile(r"\bLIB\b\s*#?\s*[:\-]?\s*(\d+)\b", re.I)


def parse_lib_number_from_text(text: str) -> Optional[int]:
    if not text:
        return None
    m = PATTERN_LIB.search(text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def find_plate_anchors(df: pd.DataFrame) -> List[Tuple[int, int, Optional[int]]]:
    """Scan the whole sheet for cells containing 'LIB#<n>' tokens.
    Returns list of (row_idx, col_idx, lib_num_or_None), deduplicated for merged-cell echoes.
    """
    anchors: List[Tuple[int, int, Optional[int]]] = []
    H, W = df.shape
    for r in range(H):
        for c in range(W):  # 扫描整张表（逐行逐列）
            v = df.iat[r, c]
            if is_empty(v):
                continue
            num = parse_lib_number_from_text(str(v))
            if num is not None:
                anchors.append((r, c, num))
    if not anchors:
        return []

    # 去重：合并相邻 2×2 范围内的重复锚点（合并单元格常见）
    anchors.sort(key=lambda t: (t[0], t[1]))
    dedup: List[Tuple[int, int, Optional[int]]] = []
    R_PAD, C_PAD = 2, 2
    for r, c, n in anchors:
        if any(abs(r - r0) <= R_PAD and abs(c - c0) <= C_PAD for r0, c0, _ in dedup):
            continue
        dedup.append((r, c, n))
    return dedup

def find_well_headers_in_window(df: pd.DataFrame, r0: int, r1: int, c0: int, c1: int) -> Optional[Tuple[int, int, List[int]]]:
    """Find row/col headers inside a window: returns (header_row_idx, row_header_col_idx, col_numbers)
    - header row must contain at least 6 numbers among 1..12 (accept 01..12)
    - row header col must contain at least 6 letters A..H
    """
    H, W = df.shape
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(H, r1)
    c1 = min(W, c1)

    header_row_idx = None
    # Find candidate header row
    for r in range(r0, r1):
        row_vals = [str(df.iat[r, c]).strip()
                    for c in range(c0, c1) if not is_empty(df.iat[r, c])]
        nums = []
        for v in row_vals:
            m = re.fullmatch(r"0?(\d{1,2})", v)
            if m:
                try:
                    n = int(m.group(1))
                    if 1 <= n <= 12:
                        nums.append(n)
                except Exception:
                    pass
        if len(set(nums)) >= 6:
            header_row_idx = r
            # Prefer the full set of 1..12 if available
            break

    if header_row_idx is None:
        return None

    # Find candidate row header column
    row_header_col_idx = None
    for c in range(c0, c1):
        col_vals = [str(df.iat[r, c]).strip()
                    for r in range(r0, r1) if not is_empty(df.iat[r, c])]
        letters = [v for v in col_vals if re.fullmatch(r"[A-Ha-h]", v)]
        if len(set(v.upper() for v in letters)) >= 6:
            row_header_col_idx = c
            break

    if row_header_col_idx is None:
        return None

    # Extract the numeric column labels from the header row within window order
    col_numbers: List[int] = []
    for c in range(row_header_col_idx + 1, c1):
        v = str(df.iat[header_row_idx, c]).strip()
        m = re.fullmatch(r"0?(\d{1,2})", v)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 12:
                col_numbers.append(n)
        elif len(col_numbers) > 0:
            # stop at first gap after we started reading numbers
            break
    # If too few columns detected, try scanning the full row within window
    if len(col_numbers) < 6:
        col_numbers = []
        for c in range(c0, c1):
            v = str(df.iat[header_row_idx, c]).strip()
            m = re.fullmatch(r"0?(\d{1,2})", v)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 12:
                    col_numbers.append(n)
    if len(col_numbers) < 6:
        return None

    return header_row_idx, row_header_col_idx, col_numbers


# --- New helper: find a concrete grid near an anchor and return its header + column mapping

def find_grid_near_anchor(df: pd.DataFrame, ar: int, ac: int) -> Optional[Tuple[int, int, List[Tuple[int,int]]]]:
    """
    Try to locate a 96-well grid (header row/col + mapping of actual sheet columns -> 1..12)
    near the given anchor (ar, ac). Returns (hr, hc, col_positions) or None.
    """
    H, W = df.shape
    # primary window
    loc = find_well_headers_in_window(
        df,
        ar, min(H, ar + ROW_WINDOW_PRIMARY),
        max(0, ac - 5), min(W, ac + COL_WINDOW_PRIMARY)
    )
    # expanded window if needed
    if not loc:
        loc = find_well_headers_in_window(
            df,
            ar, min(H, ar + ROW_WINDOW_EXPANDED),
            0, min(W, ac + COL_WINDOW_EXPANDED)
        )
    if not loc:
        return None
    hr, hc, _ = loc

    # Map columns (actual positions) to numeric labels (1..12)
    seen = set()
    col_positions: List[Tuple[int,int]] = []
    for c in range(hc + 1, min(W, hc + 1 + 24)):
        v = str(df.iat[hr, c]).strip()
        m = re.fullmatch(r"0?(\d{1,2})", v)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 12 and n not in seen:
                col_positions.append((c, n))
                seen.add(n)
        if len(seen) >= 12:
            break
    # If we didn't see enough columns, try scanning entire header row bounds
    if len(col_positions) < 6:
        seen.clear(); col_positions.clear()
        for c in range(hc + 1, min(W, hc + 1 + 24)):
            v = str(df.iat[hr, c]).strip()
            m = re.fullmatch(r"0?(\d{1,2})", v)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 12 and n not in seen:
                    col_positions.append((c, n))
                    seen.add(n)
            if len(seen) >= 12:
                break
    if len(col_positions) < 6:
        return None
    return hr, hc, col_positions


# =========================
# --- Main Pipeline -------
# =========================

def parse_blocks_from_sheet(sheet_name: str, df: pd.DataFrame) -> List[dict]:
    """
    单表（Sheet）内含多块板时的**稳定识别**：
    - 找到所有 'LIB#<n>' 行作为锚点；
    - 对每个锚点，仅在【本锚点行 到 下一个锚点行(不含)】这个“垂直带”里解析 8×12 网格；
    - 行头列（A..H）优先选离锚点列最近、且在该垂直带内出现 ≥6 个 A..H 的那一列；
    - 列头只在锚点行**右侧最多 24 列**内读取 1..12（遇到缺口即停）。
    """
    H, W = df.shape
    records: List[dict] = []

    # 1) 找到所有 LIB 锚点（行、列、编号）
    anchors = find_plate_anchors(df)
    if not anchors:
        return records
    # 按行优先、列次序排序
    anchors = sorted(anchors, key=lambda t: (t[0], t[1]))

    # 逐块解析
    for i, (r_lib, c_lib, lib_num) in enumerate(anchors):
        # 该块的垂直边界：从本锚点行开始，到下一个锚点行之前
        r_next = anchors[i + 1][0] if i + 1 < len(anchors) else H
        band_r0, band_r1 = r_lib + 1, min(H, r_next)  # 只解析 A..H 行所在的带

        # 2) 在这条带中寻找“行头列”（A..H 至少出现 6 个）
        hc_candidates = []
        for c in range(W):
            vals = []
            for rr in range(band_r0, min(band_r1, band_r0 + 20)):  # 最多向下看 20 行
                v = df.iat[rr, c]
                if is_empty(v):
                    continue
                s = str(v).strip()
                if re.fullmatch(r"[A-Ha-h]", s):
                    vals.append(s.upper())
            if len(set(vals)) >= 6:   # A..H 至少 6 个不同字母
                hc_candidates.append(c)
        if not hc_candidates:
            # 找不到行头列就跳过这一块
            continue
        # 选离 LIB 文本最近的那一列作为行头列
        hc = min(hc_candidates, key=lambda cc: abs(cc - c_lib))

        # 3) 在锚点行 r_lib 上、行头列右侧，提取 1..12 列头（遇缺口即停；只向右扫最多 24 列）
        col_positions: List[Tuple[int, int]] = []
        seen = set()
        for c in range(hc + 1, min(W, hc + 1 + 24)):
            v = str(df.iat[r_lib, c]).strip()
            m = re.fullmatch(r"0?(\d{1,2})", v)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 12 and n not in seen:
                    col_positions.append((c, n))
                    seen.add(n)
            elif col_positions:
                # 已经开始读到数字，遇到第一个缺口就停
                break
            if len(seen) >= 12:
                break
        if len(col_positions) < 6:
            # 列头太少，不像 96 孔板，跳过
            continue

        # 4) 解析 A..H 行（只在本垂直带内；只认 A..H）
        plate_label = f"LIB#{lib_num}" if lib_num is not None else sheet_name
        for rr in range(band_r0, min(band_r1, band_r0 + 16)):  # 最多向下 16 行
            row_label = str(df.iat[rr, hc]).strip()
            if not re.fullmatch(r"[A-Ha-h]", row_label):
                continue
            RL = row_label.upper()
            for c, n in col_positions:
                cell = df.iat[rr, c]
                if is_empty(cell):
                    continue
                sco_ids = extract_sco_ids(cell)
                if not sco_ids:
                    continue
                for sco in sco_ids:
                    records.append({
                        "sheet": sheet_name,
                        "plate_num": lib_num,
                        "plate_label": plate_label,
                        "well": f"{RL}{n}",
                        "SCO_id": sco,
                        "raw_value": str(cell),
                    })

    return records

def main():
    # --- Load annotation ---
    anno_df = pd.read_csv(ANNO_CSV_PATH)
    anno_df.columns = [c.strip() for c in anno_df.columns]

    # Find primary SCO column, then normalize to SCO_id
    candidate_cols = [
        "SCO_id", "SCO", "sco", "locus_tag", "Gene", "gene", "locus", "gene_id", "GeneID",
    ]
    sco_col = next((c for c in candidate_cols if c in anno_df.columns), None)
    if sco_col is None:
        for c in anno_df.columns:
            if anno_df[c].astype(str).str.contains(r"\bSCO[_-]?\d{3,5}\b", na=False, regex=True).any():
                sco_col = c
                break
        if sco_col is None:
            sco_col = anno_df.columns[0]

    anno_df["SCO_id"] = anno_df[sco_col].apply(normalize_to_sco_id)
    # De-duplicate by SCO_id to avoid 1->N merge explosions
    anno_dedup = anno_df.drop_duplicates(subset=["SCO_id"]).copy()

    # --- Load Excel ---
    try:
        xls = pd.ExcelFile(LIB_EXCEL_PATH, engine="openpyxl")
    except Exception as e:
        sys.stderr.write(f"[Error] Failed to open Excel with openpyxl: {e}\n")
        sys.stderr.write("Hint: pip install openpyxl\n")
        sys.exit(1)

    all_records: List[dict] = []
    for sheet_name in xls.sheet_names:
        raw = xls.parse(sheet_name=sheet_name, header=None)
        # First try block-based parsing (multiple plates in one sheet)
        recs = parse_blocks_from_sheet(sheet_name, raw)
        all_records.extend(recs)

    lib_long = pd.DataFrame(all_records)

    if lib_long.empty:
        sys.stderr.write(
            "[Error] No SCO ids parsed from the library map. Check the layout and that cells really contain SCO ids.\n")
        sys.exit(2)

    # Construct canonical position string
    lib_long = lib_long[~lib_long["SCO_id"].isna()].copy()
    lib_long["plate_well"] = lib_long["plate_label"].astype(
        str) + ":" + lib_long["well"].astype(str)

    # --- Aggregate positions per gene ---
    pos_agg = (
        lib_long.groupby("SCO_id")["plate_well"]
        .apply(lambda s: "; ".join(sorted(set(map(str, s)))))
        .reset_index()
        .rename(columns={"plate_well": "positions_joined"})
    )

    # --- Table 1: CRISPRi genes + positions + full annotations (deduped) ---
    table1 = pos_agg.merge(anno_dedup, on="SCO_id", how="left")
    # Reorder key columns first
    key_cols = [
        "SCO_id", "positions_joined",
        "product", "Product", "annotation", "Annotation", "description", "Description",
        "annotation direction", "Annotation direction", "annotation_direction", "direction", "Direction", "strand", "Strand",
    ]
    front = [c for c in key_cols if c in table1.columns]
    rest = [c for c in table1.columns if c not in front]
    table1 = table1[front + rest]

    # --- Table 2: full annotation + CRISPRi positions ---
    table2 = anno_df.copy()
    table2 = table2.merge(pos_agg, on="SCO_id", how="left")
    table2 = table2.rename(columns={"positions_joined": "CRISPRi_positions"})
    front2 = [c for c in ["SCO_id", "CRISPRi_positions"] if c in table2.columns]
    rest2 = [c for c in table2.columns if c not in front2]
    table2 = table2[front2 + rest2]

    # --- Save ---
    out1 = Path("CRISPRi_library_targets_with_annotations.csv")
    out2 = Path("annotation_plus_CRISPRi_positions.csv")
    table1.to_csv(out1, index=False)
    table2.to_csv(out2, index=False)

    # --- Report ---
    sheets_joined = ", ".join(xls.sheet_names)
    uniq_genes = lib_long["SCO_id"].nunique()
    uniq_positions = lib_long["plate_well"].nunique()

    print("=== Done ===")
    print(f"Sheets parsed: {sheets_joined}")
    print(
        f"Anchors found: {len(find_plate_anchors(xls.parse(xls.sheet_names[0], header=None)))} (first sheet)")
    print(f"Library rows parsed: {len(lib_long)}")
    print(f"Unique SCO ids in library: {uniq_genes}")
    print(f"Unique (plate:well) positions: {uniq_positions}")
    print(f"Output -> {out1.resolve()}")
    print(f"Output -> {out2.resolve()}")

    # Helpful hint if some plate numbers are None
    if lib_long["plate_label"].astype(str).str.match(r"^LIB#None$").any():
        print("\n[Hint] Some plate numbers were not detected.")
        print("       Precedence: MANUAL_LIB_MAP@anchor > MANUAL_LIB_MAP@sheet > LIB text in anchor cell > encounter order.")
        print("       To force numbers, add entries to MANUAL_LIB_MAP (you can use 'Sheet@row,col').")


if __name__ == "__main__":
    if not LIB_EXCEL_PATH.exists():
        sys.stderr.write(f"[Error] Missing Excel file: {LIB_EXCEL_PATH}\n")
        sys.exit(1)
    if not ANNO_CSV_PATH.exists():
        sys.stderr.write(f"[Error] Missing annotation CSV: {ANNO_CSV_PATH}\n")
        sys.exit(1)
    main()
