#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an interactive bilingual (ZH/EN) HTML explorer for a regulator→target network
with per-gene annotations. Also merges regulator info into the annotation and exports
several analysis tables (hub TFs, highly regulated genes, self-regulators, SCC modules).

Usage:
  python build_regnet_annotated_explorer.py \
    --map  Streptomyces_coelicolor_A32_regulator_to_target_analysis.tsv.txt \
    --anno scoelicolor_complete_annotation_dictionary.csv \
    --out  RegNet_Annotated_Explorer.html
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import networkx as nx


# ------------------------- Utils -------------------------
def norm_sco(x):
    """Normalize SCO-like locus tags. E.g. 'sco 4434' -> 'SCO4434' (strip NBSP/whitespace)."""
    if pd.isna(x):
        return x
    s = str(x)
    s = s.replace("\xa0", "").strip()  # remove NBSP
    s = s.replace("sco", "SCO").replace("SCo", "SCO").replace("ScO", "SCO")
    s = re.sub(r"\s+", "", s)          # collapse internal whitespaces
    return s


def load_mapping(map_path: Path) -> pd.DataFrame:
    """
    Robustly load regulator→target table.
    - detect encodings / separators
    - support multi-value targets split by comma/semicolon/whitespace
    - ensure two columns: 'regulator', 'target'
    - return unique pairs
    """
    encs = ["utf-8", "utf-8-sig", "latin1"]
    seps = ["\t", ",", ";", "|"]
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(map_path, sep=sep,
                                 encoding=enc, engine="python")
                cols = [c.lower() for c in df.columns]
                if any("reg" in c for c in cols) and any("target" in c for c in cols):
                    # rename to canonical names
                    colmap = {}
                    for c in df.columns:
                        cl = c.lower()
                        if "reg" in cl and "regulator" not in colmap.values():
                            colmap[c] = "regulator"
                        elif "target" in cl and "target" not in colmap.values():
                            colmap[c] = "target"
                    df = df.rename(columns=colmap)
                    # normalize + explode
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

    # Fallback: manual parse (lines like "REG\tT1 T2;T3")
    regs, tars = [], []
    with open(map_path, "r", encoding="utf-8", errors="ignore") as f:
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
    """Heuristics to detect the gene ID column (SCO-like)."""
    best, best_score = None, -1
    for c in df.columns:
        cl = c.strip().lower()
        score = 0
        if cl in ["sco", "sco_id", "gene", "geneid", "gene_id",
                  "locus", "locus_tag", "scoid", "sco_no", "sco_number"]:
            score += 10
        if "sco" in cl:
            score += 5
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


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True,
                    help="Regulator→target file (TSV/CSV).")
    ap.add_argument("--anno", required=True,
                    help="Annotation CSV (will be enriched).")
    ap.add_argument("--out", required=True, help="Output HTML file.")
    ap.add_argument("--id-col", default=None,
                    help="Explicit gene id column in annotation (optional).")
    args = ap.parse_args()

    map_path = Path(args.map)
    anno_path = Path(args.anno)
    out_html = Path(args.out)

    # 1) Load mapping
    rt = load_mapping(map_path)
    reg_set = set(rt["regulator"].unique())
    tgt_set = set(rt["target"].unique())
    nodes = sorted(reg_set.union(tgt_set))

    # 2) Load & normalize annotation; compute 5 regulatory columns from mapping
    anno = pd.read_csv(anno_path, encoding="utf-8")
    gene_col = args.id_col or detect_gene_col(anno)
    if gene_col is None:
        raise SystemExit("Cannot detect gene id column; please pass --id-col")

    anno["_ID_"] = anno[gene_col].map(norm_sco)

    # Build maps
    regulates = defaultdict(set)     # regulator -> set(targets)
    regulated_by = defaultdict(set)  # target   -> set(regulators)
    for r, t in rt.itertuples(index=False):
        regulates[r].add(t)
        regulated_by[t].add(r)

    # Overwrite/add 5 columns
    anno["Regulating_TFs"] = anno["_ID_"].map(
        lambda g: join_sorted(regulated_by.get(g, set())))
    anno["TF_Count"] = anno["_ID_"].map(
        lambda g: len(regulated_by.get(g, set())))
    anno["Is_TF"] = anno["_ID_"].map(lambda g: g in reg_set)
    anno["Regulates_Genes"] = anno["_ID_"].map(
        lambda g: join_sorted(regulates.get(g, set())))
    anno["Target_Count"] = anno["_ID_"].map(
        lambda g: len(regulates.get(g, set())))

    merged_csv = out_html.with_name("reg_anno_merged.csv")
    anno.drop(columns=["_ID_"]).to_csv(
        merged_csv, index=False, encoding="utf-8")

    # 3) Build node & edge elements for cytoscape
    # Pick displayed annotation columns (auto + common fields)
    display_cols = [gene_col]
    for c in ["Product", "Description", "Function", "Pathway", "BGC", "Family",
              "Is_TF", "TF_Count", "Regulating_TFs", "Target_Count", "Regulates_Genes"]:
        if c in anno.columns and c not in display_cols:
            display_cols.append(c)
    # pad some more columns up to ~12 for richer info
    if len(display_cols) < 12:
        for c in anno.columns:
            if c not in display_cols and c != "_ID_":
                display_cols.append(c)
            if len(display_cols) >= 12:
                break

    anno_small = anno.assign(_ID_=anno[gene_col].map(norm_sco))[
        ["_ID_"] + display_cols].drop_duplicates("_ID_")
    anno_dict = anno_small.set_index("_ID_").to_dict(orient="index")

    node_elements = []
    for n in nodes:
        ann = anno_dict.get(n, {})
        # merge Is_TF/Counts robustly
        is_tf = (n in reg_set)
        tf_cnt = int(ann.get("TF_Count", 0)) if ann and pd.notna(
            ann.get("TF_Count", 0)) else 0
        tgt_cnt = int(ann.get("Target_Count", 0)) if ann and pd.notna(
            ann.get("Target_Count", 0)) else 0
        if "Is_TF" in ann and not (isinstance(ann["Is_TF"], float) and pd.isna(ann["Is_TF"])):
            try:
                is_tf = bool(ann["Is_TF"]) if isinstance(ann["Is_TF"], (bool, int)) else (
                    str(ann["Is_TF"]).lower() in ["true", "yes", "1"])
            except Exception:
                pass

        ntype = "both" if (n in reg_set and n in tgt_set) else (
            "regulator_only" if n in reg_set else "target_only")
        d = {"id": n, "label": n, "type": ntype, "Is_TF": is_tf,
             "TF_Count": tf_cnt, "Target_Count": tgt_cnt}
        for k, v in (ann or {}).items():
            if k == gene_col:
                d["GeneID"] = v
            else:
                d[k] = ("" if pd.isna(v) else v)
        node_elements.append({"data": d})

    edge_elements = []
    for r, t in rt.itertuples(index=False):
        edge_elements.append({"data": {
                             "id": f"{r}->{t}", "source": r, "target": t, "isRegToReg": bool(t in reg_set)}})

    # 4) Reg→Reg SCCs (feedback modules)
    Grr = nx.from_pandas_edgelist(rt[rt["target"].isin(reg_set)],
                                  source="regulator", target="target", create_using=nx.DiGraph())
    sccs = list(nx.strongly_connected_components(Grr))
    scc_map = {}
    for i, comp in enumerate(sccs):
        for n in comp:
            scc_map[n] = i
    for ele in node_elements:
        n = ele["data"]["id"]
        ele["data"]["SCC_ID"] = int(scc_map.get(n, -1))

    # 5) Pack data for HTML
    out_map = defaultdict(list)
    in_map = defaultdict(list)
    for r, t in rt.itertuples(index=False):
        out_map[r].append(t)
        in_map[t].append(r)

    DATA = {
        "elements": {"nodes": node_elements, "edges": edge_elements},
        "outMap": {k: v for k, v in out_map.items()},
        "inMap": {k: v for k, v in in_map.items()},
        "regulators": sorted(list(reg_set)),
        "displayCols": display_cols
    }

    i18n = {
        "zh": {
            "title": "Annotated RegNet Explorer — 带注释的调控网络",
            "subtitle": "内嵌注释信息，可检索/过滤/导出",
            "search": "搜索/选择节点",
            "show_all": "显示全网",
            "ego": "1跳邻域",
            "filters": "过滤与布局",
            "only_rr": "只看 Reg→Reg",
            "is_tf": "仅显示 TF",
            "tf_min": "TF_Count ≥ ",
            "tgt_min": "Target_Count ≥ ",
            "layout": "布局",
            "apply": "应用",
            "path": "最短路径",
            "go": "查找",
            "export_png": "导出图为 PNG",
            "anno": "注释信息",
            "up": "上游（所有 TF）",
            "down": "下游（所有 targets）",
            "export_csv": "导出 CSV"
        },
        "en": {
            "title": "Annotated RegNet Explorer — Regulatory Network with Annotations",
            "subtitle": "Embedded annotations with search/filter/export",
            "search": "Search/Select Node",
            "show_all": "Show Full Network",
            "ego": "1-hop Ego",
            "filters": "Filters & Layout",
            "only_rr": "Reg→Reg only",
            "is_tf": "Show TF only",
            "tf_min": "TF_Count ≥ ",
            "tgt_min": "Target_Count ≥ ",
            "layout": "Layout",
            "apply": "Apply",
            "path": "Shortest Path",
            "go": "Find",
            "export_png": "Export PNG",
            "anno": "Annotation",
            "up": "Upstream (all TFs)",
            "down": "Downstream (all targets)",
            "export_csv": "Export CSV"
        }
    }

    # 6) HTML template
    template = r"""
<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>Annotated RegNet Explorer</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
#app{display:grid;grid-template-columns:360px 1fr;grid-template-rows:auto 1fr auto;height:100vh}
header{grid-column:1/3;display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-bottom:1px solid #eee;background:#fafafa}
#sidebar{padding:10px;border-right:1px solid #eee;overflow:auto}
#main{display:grid;grid-template-rows:60% 40%}
#cy{position:relative;height:100%;border-bottom:1px solid #eee}
#cy>div{position:absolute;inset:0}
#detail{position:relative}
#cy2{position:absolute;inset:0}
#tables{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #eee}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{border-bottom:1px solid #eee;padding:6px 8px;text-align:left;vertical-align:top}
button{padding:6px 10px;border:1px solid #ddd;background:#fff;border-radius:8px;cursor:pointer;margin-right:6px}
.tiny{font-size:12px;color:#666}
input,select{padding:6px 8px;border:1px solid #ddd;border-radius:8px;width:100%}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.badge{display:inline-block;padding:2px 6px;border-radius:6px;background:#eef;margin-right:4px}
</style>
</head><body>
<div id="app">
<header>
  <div><b id="t-title"></b><div class="tiny" id="t-sub"></div></div>
  <div><select id="lang"><option value="zh">中文</option><option value="en">English</option></select></div>
</header>
<aside id="sidebar">
  <div class="tiny" id="t-search"></div>
  <input id="q" list="nodes" placeholder="SCOxxxx"/><datalist id="nodes"></datalist>
  <div style="margin:8px 0">
    <button id="btnAll"></button>
    <button id="btnEgo"></button>
  </div>
  <div class="tiny" id="t-filters" style="margin-top:6px"></div>
  <div>
    <button id="btnRR"></button>
    <label class="tiny"><input type="checkbox" id="chkTF"/> <span id="t-istf"></span></label>
  </div>
  <div class="grid" style="margin-top:6px">
    <div class="tiny"><span id="t-tfmin"></span><b id="tfv">0</b></div>
    <input type="range" id="tfmin" min="0" max="50" step="1" value="0"/>
    <div class="tiny"><span id="t-tgtmin"></span><b id="tgtv">0</b></div>
    <input type="range" id="tgtmin" min="0" max="100" step="1" value="0"/>
  </div>
  <div class="tiny" id="t-path" style="margin-top:6px">Shortest Path / 最短路径</div>
  <input id="src" list="nodes" placeholder="Source"/><input id="dst" list="nodes" placeholder="Target"/>
  <div style="margin-top:6px"><button id="go"></button> <button id="png"></button></div>
  <h4 style="margin:10px 0" id="t-anno"></h4>
  <div id="info" class="tiny">—</div>
</aside>
<main id="main">
  <div id="cy"><div id="cyview"></div></div>
  <div id="detail"><div id="cy2"></div></div>
</main>
<section id="tables">
  <div style="padding:8px"><h3 id="t-up"></h3>
    <button id="expUp"></button>
    <table id="tblUp"><thead><tr><th>Regulator</th></tr></thead><tbody></tbody></table>
  </div>
  <div style="padding:8px"><h3 id="t-down"></h3>
    <button id="expDown"></button>
    <table id="tblDown"><thead><tr><th>Target</th></tr></thead><tbody></tbody></table>
  </div>
</section>
</div>
<script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
<script>
const I18N = %%I18N%%;
const DATA = %%DATA%%;
let LANG = 'zh';

function t(k){ return (I18N[LANG] && I18N[LANG][k]) || I18N.en[k] || k; }
function setTexts(){
  document.getElementById('t-title').textContent = t('title');
  document.getElementById('t-sub').textContent = t('subtitle');
  document.getElementById('t-search').textContent = t('search');
  document.getElementById('btnAll').textContent = t('show_all');
  document.getElementById('btnEgo').textContent = t('ego');
  document.getElementById('t-filters').textContent = t('filters');
  document.getElementById('btnRR').textContent = t('only_rr');
  document.getElementById('t-istf').textContent = t('is_tf');
  document.getElementById('t-tfmin').textContent = t('tf_min');
  document.getElementById('t-tgtmin').textContent = t('tgt_min');
  document.getElementById('apply').textContent = t('apply') || 'Apply';
  document.getElementById('t-anno').textContent = t('anno');
  document.getElementById('t-up').textContent = t('up');
  document.getElementById('t-down').textContent = t('down');
  document.getElementById('expUp').textContent = t('export_csv');
  document.getElementById('expDown').textContent = t('export_csv');
  document.getElementById('go').textContent = t('go');
  document.getElementById('png').textContent = t('export_png');
}
document.getElementById('lang').onchange = (e)=>{ LANG = e.target.value; setTexts(); };
setTexts();

// datalist for node IDs
const dl = document.getElementById('nodes');
DATA.elements.nodes.forEach(n=>{ const o=document.createElement('option'); o.value=n.data.id; dl.appendChild(o); });

// base style
const baseStyle=[
 { selector:'node', style:{ 'width':14,'height':14,'label':'data(label)','font-size':9,'text-valign':'center','text-halign':'center','background-color':'#888','color':'#222' }},
 { selector:'node[type = "regulator_only"]', style:{'background-color':'#6fa8dc'} },
 { selector:'node[type = "target_only"]', style:{'background-color':'#f9cb9c'} },
 { selector:'node[type = "both"]', style:{'background-color':'#93c47d'} },
 { selector:'edge', style:{ 'curve-style':'bezier','width':0.8,'line-color':'#bbb','target-arrow-shape':'triangle','target-arrow-color':'#bbb' } },
 { selector:'edge[isRegToReg = "false"]', style:{ 'line-style':'dashed','opacity':0.6 } },
 { selector:'.fade', style:{ 'opacity':0.15 } },
 { selector:'.hi', style:{ 'line-color':'#ff6666','target-arrow-color':'#ff6666','width':2 } },
 { selector:'edge[source = target]', style:{ 'curve-style':'bezier','control-point-step-size':30,'loop-direction':'-45deg','loop-sweep':'30deg','width':1.2,'line-color':'#999' } }
];

const cy = cytoscape({ container: document.getElementById('cyview'),
  elements: DATA.elements, style: baseStyle, layout:{name:'cose'}, wheelSensitivity:0.2 });
const cy2 = cytoscape({ container: document.getElementById('cy2'),
  elements: [], style: baseStyle, layout:{name:'concentric'}, wheelSensitivity:0.2 });

function info(id){
  const n = cy.$(`node[id = "${id}"]`);
  if (n.length===0){ document.getElementById('info').textContent='—'; return; }
  const d = n.data();
  const keys = DATA.displayCols;
  let html = `<div><b>${d.label}</b></div>`;
  html += `<div class="tiny">Is_TF: ${d.Is_TF} | TF_Count: ${d.TF_Count} | Target_Count: ${d.Target_Count} | SCC_ID: ${d.SCC_ID ?? '-'}</div>`;
  html += `<div style="margin-top:6px">`;
  keys.forEach(k=>{
    const v = (d[k]===undefined || d[k]==="") ? "—" : d[k];
    html += `<div><span class="badge">${k}</span> ${v}</div>`;
  });
  html += `</div>`;
  document.getElementById('info').innerHTML = html;
}
function updateTables(id){
  const up = DATA.inMap[id] || [];
  const down = DATA.outMap[id] || [];
  const tbU = document.querySelector('#tblUp tbody'); tbU.innerHTML='';
  const tbD = document.querySelector('#tblDown tbody'); tbD.innerHTML='';
  up.slice().sort().forEach(g=>{ tbU.insertAdjacentHTML('beforeend', `<tr><td>${g}</td></tr>`); });
  down.slice().sort().forEach(g=>{ tbD.insertAdjacentHTML('beforeend', `<tr><td>${g}</td></tr>`); });
}
function exportTableCSV(tblId, filename){
  const rows = Array.from(document.querySelectorAll(`#${tblId} tr`));
  const csv = rows.map(r => Array.from(r.querySelectorAll('th,td')).map(td => `"${td.textContent.replace(/"/g,'""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'}); const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click(); URL.revokeObjectURL(url);
}
document.getElementById('expUp').onclick = ()=> exportTableCSV('tblUp','upstream.csv');
document.getElementById('expDown').onclick = ()=> exportTableCSV('tblDown','downstream.csv');

function selectNode(id){
  const n = cy.$(`node[id = "${id}"]`); if (n.length===0) return;
  const neigh = n.closedNeighborhood();
  cy.$('.fade').removeClass('fade'); cy.$('.hi').removeClass('hi');
  cy.elements().difference(neigh).addClass('fade');
  info(id); updateTables(id);
  cy.fit(neigh, 80);
  cy2.elements().remove(); cy2.add(neigh.jsons()); cy2.layout({name:'breadthfirst', directed:true}).run(); cy2.fit(cy2.elements(), 60);
}

function applyFilters(){
  const rr = window._onlyRR || false;
  const tfOnly = document.getElementById('chkTF').checked;
  const tfMin = parseInt(document.getElementById('tfmin').value);
  const tgtMin = parseInt(document.getElementById('tgtmin').value);
  cy.batch(()=>{
    cy.$('.fade').removeClass('fade');
    if (rr){ cy.$('edge[isRegToReg = "false"]').addClass('fade'); }
    cy.nodes().forEach(n=>{
      const isTF = !!n.data('Is_TF');
      const tfc = n.data('TF_Count') || 0;
      const tgc = n.data('Target_Count') || 0;
      let hide = false;
      if (tfOnly && !isTF) hide = true;
      if (tfc < tfMin) hide = true;
      if (tgc < tgtMin) hide = true;
      if (hide) n.addClass('fade');
    });
  });
}

document.getElementById('btnAll').onclick = ()=>{ cy.$('.fade').removeClass('fade'); cy.$('.hi').removeClass('hi'); cy.fit(cy.elements(), 50); };
document.getElementById('btnEgo').onclick = ()=>{ const id=document.getElementById('q').value.trim(); if(id) selectNode(id); };
document.getElementById('btnRR').onclick = ()=>{ window._onlyRR = !window._onlyRR; applyFilters(); };
document.getElementById('tfmin').oninput = (e)=>{ document.getElementById('tfv').textContent=e.target.value; applyFilters(); };
document.getElementById('tgtmin').oninput = (e)=>{ document.getElementById('tgtv').textContent=e.target.value; applyFilters(); };
document.getElementById('chkTF').onchange = ()=> applyFilters();

document.getElementById('go').onclick = ()=>{
  const s=document.getElementById('src').value.trim();
  const d=document.getElementById('dst').value.trim();
  if(!s||!d) return;
  // BFS shortest path on directed outMap
  const Q=[s]; const prev=new Map(); prev.set(s,null);
  while(Q.length){
    const x=Q.shift();
    for(const y of (DATA.outMap[x]||[])){
      if(!prev.has(y)){ prev.set(y,x); Q.push(y); }
      if(y===d){ Q.length=0; break; }
    }
  }
  if(!prev.has(d)) return;
  const path=[]; let cur=d; while(cur!==null){ path.push(cur); cur=prev.get(cur); } path.reverse();
  cy.$('.hi').removeClass('hi'); cy.$('.fade').removeClass('fade');
  for(let i=0;i<path.length;i++){
    const n = cy.$(`node[id = "${path[i]}"]`); n.removeClass('fade');
    if(i<path.length-1){ const e = cy.$(`edge[source = "${path[i]}"][target = "${path[i+1]}"]`); e.addClass('hi'); }
  }
  const eles = cy.collection(path.map(id=>cy.$(`node[id = "${id}"]`))).union(cy.$('.hi'));
  cy.fit(eles, 80);
};

document.getElementById('png').onclick = ()=>{
  const a=document.createElement('a'); a.href=cy.png({full:false, scale:2, bg:'#fff'}); a.download='regnet_annotated.png'; a.click();
};

cy.on('tap','node',evt=>{ document.getElementById('q').value = evt.target.id(); selectNode(evt.target.id()); });

// init
cy.fit(cy.elements(), 60);
</script>
</body></html>
"""

    html = (template
            .replace("%%I18N%%", json.dumps(i18n, ensure_ascii=False))
            .replace("%%DATA%%", json.dumps(DATA, ensure_ascii=False))
            )
    out_html.write_text(html, encoding="utf-8")

    # 7) Analyses & tables
    # hub TFs
    anno2 = pd.read_csv(merged_csv, encoding="utf-8")
    gcol = gene_col
    anno2["_ID_"] = anno2[gcol].map(norm_sco)
    hub_df = (anno2[["_ID_", gcol, "Is_TF", "Target_Count", "Regulates_Genes"]]
              .drop_duplicates("_ID_"))
    hub_df = hub_df[hub_df["Is_TF"] == True].sort_values(
        "Target_Count", ascending=False)
    hub_df.rename(columns={gcol: "GeneID"}, inplace=True)
    hub_df.drop(columns=["_ID_"], inplace=True)
    hub_df.to_csv(out_html.with_name(
        "top_hub_TFs_by_TargetCount.csv"), index=False)

    # highly regulated
    hr_df = (anno2[["_ID_", gcol, "TF_Count", "Regulating_TFs"]]
             .drop_duplicates("_ID_")
             .sort_values("TF_Count", ascending=False))
    hr_df.rename(columns={gcol: "GeneID"}, inplace=True)
    hr_df.drop(columns=["_ID_"], inplace=True)
    hr_df.to_csv(out_html.with_name(
        "top_highly_regulated_genes_by_TFCount.csv"), index=False)

    # self regulators
    rt_set = set(map(tuple, rt[["regulator", "target"]].values.tolist()))
    auto = sorted([r for r in reg_set if (r, r) in rt_set])
    pd.DataFrame({"GeneID": auto}).to_csv(
        out_html.with_name("self_regulators.csv"), index=False)

    # SCC summary
    scc_list = []
    for i, comp in enumerate(sccs):
        for n in comp:
            scc_list.append({"SCC_ID": i, "GeneID": n, "Size": len(comp)})
    scc_df = pd.DataFrame(scc_list).sort_values(
        ["Size", "SCC_ID"], ascending=[False, True])
    scc_df.to_csv(out_html.with_name("regulator_SCC_summary.csv"), index=False)

    print(f"[OK] HTML: {out_html}")
    print(f"[OK] Merged annotation: {merged_csv}")
    print(f"[OK] Tables: ",
          out_html.with_name("top_hub_TFs_by_TargetCount.csv"),
          out_html.with_name("top_highly_regulated_genes_by_TFCount.csv"),
          out_html.with_name("self_regulators.csv"),
          out_html.with_name("regulator_SCC_summary.csv"),
          sep="\n  - ")


if __name__ == "__main__":
    main()
