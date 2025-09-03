#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge regulator→target relationships into an annotation table.

What it does:
- Robustly load the regulator→target file and **split+explode** multi-target cells.
- Normalize IDs (sco→SCO, strip weird spaces).
- Build maps:
    regulated_by[gene] = set(regulators)
    regulates[regulator] = set(targets)
- Overwrite / create these 5 columns in the annotation:
    Regulating_TFs, TF_Count, Is_TF, Regulates_Genes, Target_Count
- Write comma-separated strings (no brackets), and correct counts.

Usage:
    python merge_regulatory_into_annotation.py \
        --anno scoelicolor_complete_annotation_dictionary.csv \
        --map  Streptomyces_coelicolor_A32_regulator_to_target_analysis.tsv.txt \
        --out  scoelicolor_annotation_with_regulatory_links_v2.csv
"""

import argparse
import re
from pathlib import Path
import pandas as pd
from collections import defaultdict


def norm_sco(x):
    if pd.isna(x):
        return x
    s = str(x)
    # remove non-breaking spaces
    s = s.replace("\xa0", "").strip()
    s = s.replace("sco", "SCO").replace("SCo", "SCO").replace("ScO", "SCO")
    # collapse internal whitespaces
    s = re.sub(r"\s+", "", s)
    return s


def load_reg_to_target(p: Path) -> pd.DataFrame:
    """Robust loader that SPLITS multi-target cells and explodes to one target per row."""
    encs = ["utf-8", "utf-8-sig", "latin1"]
    seps = ["\t", ",", ";", "|"]
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(p, sep=sep, encoding=enc, engine="python")
                cols = [c.lower() for c in df.columns]
                if any("reg" in c for c in cols) and any("target" in c for c in cols):
                    # rename the two key columns
                    colmap = {}
                    for c in df.columns:
                        cl = c.lower()
                        if "reg" in cl and "regulator" not in colmap.values():
                            colmap[c] = "regulator"
                        elif "target" in cl and "target" not in colmap.values():
                            colmap[c] = "target"
                    df = df.rename(columns=colmap)

                    # normalize & EXPLODE targets
                    df["regulator"] = df["regulator"].map(norm_sco)
                    df["target"] = df["target"].astype(str)
                    df = df.assign(target=df["target"].str.split(
                        r"[;,\s]+")).explode("target")
                    df["target"] = df["target"].map(norm_sco)

                    df = df.dropna(subset=["regulator", "target"])
                    df = df[(df["regulator"] != "") & (df["target"] != "")]
                    return df[["regulator", "target"]].drop_duplicates()
            except Exception:
                continue

    # fallback: manual line parse
    regs, tars = [], []
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "\t" in s:
                reg, rest = s.split("\t", 1)
            elif ":" in s:
                reg, rest = s.split(":", 1)
            else:
                parts = re.split(r"\s+", s, maxsplit=1)
                if len(parts) != 2:
                    continue
                reg, rest = parts
            for t in re.split(r"[;,\s]+", rest.strip()):
                t = t.strip()
                if t:
                    regs.append(norm_sco(reg))
                    tars.append(norm_sco(t))
    return pd.DataFrame({"regulator": regs, "target": tars}).drop_duplicates()


def detect_gene_col(df: pd.DataFrame) -> str:
    """Heuristics to find the gene ID column in the annotation table."""
    best, best_score = None, -1
    for c in df.columns:
        cl = c.strip().lower()
        score = 0
        if cl in ["sco", "sco_id", "gene", "geneid", "gene_id", "locus", "locus_tag", "scoid", "sco_no", "sco_number"]:
            score += 10
        if "sco" in cl:
            score += 5
        if "locus" in cl:
            score += 3
        try:
            series = df[c].astype(str).fillna("")
            hit = series.str.contains(
                r"^S?CO\d{3,5}$", case=False, regex=True).mean()
            score += hit * 10
        except Exception:
            pass
        if score > best_score:
            best_score, best = score, c
    return best


def join_sorted(values) -> str:
    if not values:
        return ""
    return ",".join(sorted(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", required=True, help="Annotation CSV")
    ap.add_argument("--map",  required=True,
                    help="Regulator→target mapping file (TSV/CSV; targets may be comma-separated)")
    ap.add_argument("--out",  required=True, help="Output CSV")
    args = ap.parse_args()

    anno = pd.read_csv(args.anno, encoding="utf-8")
    gene_col = detect_gene_col(anno)
    anno["_SCO_norm_"] = anno[gene_col].map(norm_sco)

    reg_df = load_reg_to_target(Path(args.map))

    # Build maps using sets
    from collections import defaultdict
    regulates = defaultdict(set)  # regulator -> set(targets)
    regulated_by = defaultdict(set)  # gene -> set(regulators)
    for r, t in reg_df.itertuples(index=False):
        regulates[r].add(t)
        regulated_by[t].add(r)

    # Unique regulator set for Is_TF
    reg_set = set(reg_df["regulator"].unique())

    # Overwrite / add five columns (strings without brackets)
    anno["Regulating_TFs"] = anno["_SCO_norm_"].map(
        lambda g: join_sorted(regulated_by.get(g, set())))
    anno["TF_Count"] = anno["_SCO_norm_"].map(
        lambda g: len(regulated_by.get(g, set())))
    anno["Is_TF"] = anno["_SCO_norm_"].map(lambda g: g in reg_set)
    anno["Regulates_Genes"] = anno["_SCO_norm_"].map(
        lambda g: join_sorted(regulates.get(g, set())))
    anno["Target_Count"] = anno["_SCO_norm_"].map(
        lambda g: len(regulates.get(g, set())))

    # Save
    anno.drop(columns=["_SCO_norm_"]).to_csv(
        args.out, index=False, encoding="utf-8")
    print(
        f"Done.\n- Detected gene id column: {gene_col}\n- Regulators: {len(reg_set)}\n- Edges: {reg_df.shape[0]}\n- Output: {args.out}")


df = pd.read_csv("scoelicolor_annotation_with_regulatory_links_v2.csv")
# 1) 随机看 5 行
print(df[["GeneID", "Is_TF", "TF_Count", "Target_Count",
      "Regulating_TFs", "Regulates_Genes"]].sample(5, random_state=0))
# 2) 看最受调控的基因（TF_Count 最大）
print(df.nlargest(10, "TF_Count")[["GeneID", "TF_Count", "Regulating_TFs"]])
# 3) 看最“能调”的 TF（Target_Count 最大）
print(df.nlargest(10, "Target_Count")[
      ["GeneID", "Target_Count", "Regulates_Genes"]])
# 4) 自调控（既是 TF 又受自己调控）
self_reg = df[(df["Is_TF"]) & df.apply(lambda r: str(r["GeneID"])
                                       in str(r["Regulating_TFs"]).split(","), axis=1)]
print(self_reg[["GeneID", "Regulating_TFs", "Regulates_Genes"]].head())

if __name__ == "__main__":
    main()
