#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a bilingual interactive HTML explorer (CN/EN) for a regulator→target mapping table.

Features in the generated HTML:
- Node search & selection
- Ego network (1-hop)
- Outward tree / Inward tree (depth-configurable)
- Shortest path finder between any two nodes
- Toggle all edges vs. Reg→Reg-only
- Degree filter (out-degree to other regulators)
- Multiple layouts (cose, concentric, circle, breadthfirst)
- Node info card (type, in/out degree wrt regulators, SCC id)
- Upstream/downstream tables with CSV export
- Export current view as PNG

Usage:
    python build_regnet_explorer_bilingual.py \
        --input Streptomyces_coelicolor_A32_regulator_to_target_analysis.tsv.txt \
        --output RegNet_Explorer_Bilingual.html
"""

import argparse
from pathlib import Path
import pandas as pd
import re
import json
import networkx as nx


def load_mapping(path: Path) -> pd.DataFrame:
    """
    Robustly load regulator→target mapping with unknown separators/encodings.
    Expected columns: *something* like "regulator" and "target".
    If targets are grouped in a cell, we explode them by common delimiters.
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1"]
    seps_to_try = ["\t", ",", ";", "|"]

    for enc in encodings_to_try:
        for sep in seps_to_try:
            try:
                tmp = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                cols = [c.lower() for c in tmp.columns]
                if any("reg" in c for c in cols) and any("target" in c for c in cols):
                    df = tmp.copy()
                    # normalize column names
                    colmap = {}
                    for c in df.columns:
                        cl = c.lower()
                        if "reg" in cl and "regulator" not in colmap.values():
                            colmap[c] = "regulator"
                        elif "target" in cl and "target" not in colmap.values():
                            colmap[c] = "target"
                    df = df.rename(columns=colmap)
                    # explode target list if needed
                    if "target" in df.columns:
                        df["target"] = df["target"].astype(str)
                        if df["target"].str.contains(r"[;, ]").any():
                            df = df.assign(target=df["target"].str.split(r"[;,\s]+")).explode("target")
                    return df[["regulator", "target"]]
            except Exception:
                continue

    # Fallback: manual parse lines like "reg \t t1 t2 t3" or "reg: t1, t2"
    regs, tars = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                reg, rest = line.split("\t", 1)
            elif ":" in line:
                reg, rest = line.split(":", 1)
            else:
                parts = re.split(r"\s+", line, maxsplit=1)
                if len(parts) == 2:
                    reg, rest = parts
                else:
                    # skip
                    continue
            tlist = re.split(r"[,\s;]+", rest.strip())
            for t in tlist:
                if t:
                    regs.append(reg.strip())
                    tars.append(t.strip())
    return pd.DataFrame({"regulator": regs, "target": tars})


def norm_sco(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x).strip()
    s = s.replace("sco", "SCO").replace("SCo", "SCO").replace("ScO", "SCO")
    return s


def build_html(df: pd.DataFrame, output_path: Path):
    # Clean and normalize
    df = df.copy()
    df["regulator"] = df["regulator"].map(norm_sco)
    df["target"] = df["target"].map(norm_sco)
    df = df.dropna(subset=["regulator", "target"])
    df = df[(df["regulator"] != "") & (df["target"] != "")]
    edges_all = df[["regulator", "target"]].drop_duplicates()

    # Regulators set (from "regulator" column)
    regulators = set(edges_all["regulator"].unique())

    # Reg→Reg subgraph
    edges_rr = edges_all[edges_all["target"].isin(regulators)].copy()
    edges_rr = edges_rr[edges_rr["regulator"] != edges_rr["target"]]
    Grr = nx.from_pandas_edgelist(edges_rr, source="regulator", target="target", create_using=nx.DiGraph())

    out_deg = dict(Grr.out_degree())
    in_deg = dict(Grr.in_degree())

    # SCC mapping
    sccs = list(nx.strongly_connected_components(Grr))
    node_to_scc = {}
    for i, comp in enumerate(sccs):
        for n in comp:
            node_to_scc[n] = i

    # Build nodes/edges payload
    nodes_set = set(edges_all["regulator"]).union(set(edges_all["target"]))
    node_records = []
    targets_set = set(edges_all["target"])

    for n in sorted(nodes_set):
        is_reg = n in regulators
        in_targets = n in targets_set
        if is_reg and in_targets:
            ntype = "both"
        elif is_reg:
            ntype = "regulator_only"
        else:
            ntype = "target_only"

        node_records.append({
            "data": {
                "id": n,
                "label": n,
                "type": ntype,
                "out_to_reg": int(out_deg.get(n, 0)),
                "in_from_reg": int(in_deg.get(n, 0)),
                "scc": int(node_to_scc.get(n, -1))
            }
        })

    edge_records = []
    for r, t in edges_all.itertuples(index=False):
        edge_records.append({
            "data": {
                "id": f"{r}->{t}",
                "source": r,
                "target": t,
                "isRegToReg": bool(t in regulators)
            }
        })

    # Precompute adjacency for JS (fast UI)
    out_map, in_map = {}, {}
    for r, t in edges_all.itertuples(index=False):
        out_map.setdefault(r, []).append(t)
        in_map.setdefault(t, []).append(r)

    data = {
        "elements": {"nodes": node_records, "edges": edge_records},
        "outMap": out_map,
        "inMap": in_map,
        "regulators": sorted(list(regulators))
    }

    # i18n dictionary
    i18n = {
        "zh": {
            "title": "RegNet Explorer — S. coelicolor 调控网络交互可视化",
            "subtitle": "基于你的 regulator→target 映射数据（本页内嵌，无需导入数据）",
            "search_label": "选择/搜索节点",
            "buttons_show_all": "显示全网",
            "buttons_ego": "显示 1 跳邻域",
            "buttons_out_tree": "向外扩展树",
            "buttons_in_tree": "向内追溯树",
            "depth": "树深度",
            "path_label": "路径查找（最短路径）",
            "path_src": "源节点（SCOxxxx）",
            "path_dst": "目标节点（SCOxxxx）",
            "path_go": "查找并高亮路径",
            "filters_layout": "过滤与布局",
            "edge_filters": "边过滤：",
            "all_edges": "全部边",
            "regreg_edges": "只看 Reg→Reg",
            "deg_filter": "只显示对其他 regulator 的出度 ≥ ",
            "layout": "布局：",
            "apply_layout": "应用布局",
            "export_png": "导出当前视图 PNG",
            "node_info": "节点信息",
            "node_type": "类型",
            "node_outdeg": "对 regulator 的出度",
            "node_indeg": "来自 regulator 的入度",
            "node_scc": "SCC（强连通分量）ID",
            "downstream": "该节点的 下游（所有 target）",
            "upstream": "该节点的 上游（所有 regulator）",
            "is_reg": "是否为Regulator",
            "yes": "是",
            "no": "否",
            "export_csv_out": "导出下游表 CSV",
            "export_csv_in": "导出上游表 CSV",
            "legend_nodes": "节点颜色：regulator_only / target_only / both",
            "legend_edges": "边样式：Reg→Reg 实线；Reg→Target 虚线",
            "footer": "提示：选择节点后可切换 1跳邻域、向外/向内树（可设深度），或查找任意两节点的最短路径。支持只看 Reg→Reg、出度阈值过滤和多种布局。",
            "not_found_path": "未找到从 {s} 到 {d} 的有向路径"
        },
        "en": {
            "title": "RegNet Explorer — Interactive Regulatory Network for S. coelicolor",
            "subtitle": "Built from your regulator→target mapping (data embedded)",
            "search_label": "Select/Search Node",
            "buttons_show_all": "Show Full Network",
            "buttons_ego": "Show 1-hop Neighborhood",
            "buttons_out_tree": "Expand Outward Tree",
            "buttons_in_tree": "Trace Inward Tree",
            "depth": "Tree Depth",
            "path_label": "Path Finder (Shortest Path)",
            "path_src": "Source node (SCOxxxx)",
            "path_dst": "Target node (SCOxxxx)",
            "path_go": "Find & Highlight Path",
            "filters_layout": "Filters & Layout",
            "edge_filters": "Edge filter:",
            "all_edges": "All edges",
            "regreg_edges": "Reg→Reg only",
            "deg_filter": "Show nodes with out-degree to regulators ≥ ",
            "layout": "Layout:",
            "apply_layout": "Apply",
            "export_png": "Export PNG",
            "node_info": "Node Info",
            "node_type": "Type",
            "node_outdeg": "Out-degree to regulators",
            "node_indeg": "In-degree from regulators",
            "node_scc": "SCC (Strongly Connected Component) ID",
            "downstream": "Downstream (all targets)",
            "upstream": "Upstream (all regulators)",
            "is_reg": "Is Regulator",
            "yes": "Yes",
            "no": "No",
            "export_csv_out": "Export Downstream CSV",
            "export_csv_in": "Export Upstream CSV",
            "legend_nodes": "Node colors: regulator_only / target_only / both",
            "legend_edges": "Edges: Reg→Reg solid; Reg→Target dashed",
            "footer": "Tip: After selecting a node, switch to 1-hop neighborhood, outward/inward trees (set depth), or find shortest paths. Supports Reg→Reg-only view, degree threshold filter, and multiple layouts.",
            "not_found_path": "No directed path found from {s} to {d}"
        }
    }

    # HTML template with placeholders (avoid f-string curlies issues)
    template = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>%%TITLE%%</title>
<style>
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }
  #app { display: grid; grid-template-columns: 340px 1fr; grid-template-rows: auto 1fr auto; height: 100vh; }
  header { grid-column: 1 / span 2; padding: 10px 16px; border-bottom: 1px solid #eee; background: #fafafa; display:flex; align-items:center; justify-content:space-between; gap:12px; }
  .sub { color:#777; font-size:12px; }
  #sidebar { padding: 12px; overflow: auto; border-right: 1px solid #eee; }
  #main { display: grid; grid-template-rows: 60% 40%; }
  #graph { border-bottom: 1px solid #eee; position:relative; }
  #cy { position:absolute; inset:0; }
  #detail { position:relative; }
  #cy-detail { position:absolute; inset:0; }
  #tables { display: grid; grid-template-columns: 1fr 1fr; overflow: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; }
  #footer { grid-column: 1 / span 2; padding: 8px 16px; border-top: 1px solid #eee; background: #fafafa; font-size: 12px; color: #666; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f0f3ff; margin-left: 6px; font-size: 12px; }
  .controls label { display: block; margin: 8px 0 4px; font-weight: 600; }
  input[type="text"], select { width: 100%; padding: 6px 8px; border: 1px solid #ddd; border-radius: 8px; }
  button { padding: 6px 10px; border: 1px solid #ddd; background: white; border-radius: 8px; cursor: pointer; margin-right: 6px; }
  button:hover { background: #f7f7f7; }
  .row { margin: 10px 0; }
  .chip { padding: 2px 6px; border-radius: 6px; background: #eef; margin-right: 6px; font-size: 12px; }
  .muted { color: #999; }
  .tiny { font-size: 11px; }
  .inline { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
</style>
</head>
<body>
<div id="app">
  <header>
    <div>
      <div id="hdr-title"><strong>%%TITLE%%</strong></div>
      <div class="sub" id="hdr-sub">%%SUBTITLE%%</div>
    </div>
    <div class="inline">
      <label for="langSel" class="tiny muted">Language</label>
      <select id="langSel">
        <option value="zh">中文</option>
        <option value="en">English</option>
      </select>
    </div>
  </header>
  <aside id="sidebar">
    <div class="controls">
      <label id="lbl-search">%%SEARCH_LABEL%%</label>
      <input type="text" id="nodeSearch" list="nodeList" placeholder="SCO0110 / SCO5261 / redD …" />
      <datalist id="nodeList"></datalist>
      <div class="row">
        <button id="btnShowAll">%%BTN_SHOWALL%%</button>
        <button id="btnEgo">%%BTN_EGO%%</button>
      </div>
      <div class="row">
        <label id="lbl-tree">%%FILTERS_LAYOUT%%</label>
        <div class="inline">
          <button id="btnOutTree">%%BTN_OUTTREE%%</button>
          <button id="btnInTree">%%BTN_INTREE%%</button>
          <span id="depthText" class="tiny muted">%%DEPTH%%: <b>3</b></span>
        </div>
        <input type="range" id="depthSlider" min="1" max="6" step="1" value="3" />
      </div>
      <div class="row">
        <label id="lbl-path">%%PATH_LABEL%%</label>
        <input type="text" id="pathSrc" list="nodeList" placeholder="%%PATH_SRC%%" />
        <input type="text" id="pathDst" list="nodeList" placeholder="%%PATH_DST%%" />
        <div style="margin-top:6px">
          <button id="btnPath">%%PATH_GO%%</button>
        </div>
      </div>
      <div class="row">
        <label id="lbl-filters">%%FILTERS_LAYOUT%%</label>
        <div class="tiny" id="lbl-edgefilters">%%EDGE_FILTERS%%</div>
        <button id="btnAllEdges">%%ALL_EDGES%%</button>
        <button id="btnRegRegEdges">%%REGREG_EDGES%%</button>
        <div class="tiny" style="margin-top:6px" id="lbl-degfilter">%%DEG_FILTER%%<span id="degVal">0</span></div>
        <input type="range" id="degSlider" min="0" max="80" step="1" value="0" />
        <div class="tiny" style="margin-top:6px" id="lbl-layout">%%LAYOUT%%</div>
        <select id="layoutSel">
          <option value="cose">cose</option>
          <option value="concentric">concentric</option>
          <option value="circle">circle</option>
          <option value="breadthfirst">breadthfirst</option>
        </select>
        <button id="btnLayout" style="margin-top:6px">%%APPLY_LAYOUT%%</button>
        <div class="row"><button id="btnPNG">%%EXPORT_PNG%%</button></div>
      </div>
      <div class="row">
        <h4 id="nodeInfoTitle" style="margin:8px 0;">%%NODE_INFO%%</h4>
        <div id="nodeInfo" class="tiny muted">—</div>
      </div>
      <div class="row">
        <div class="tiny muted" id="legendNodes">%%LEGEND_NODES%%</div>
        <div class="tiny muted" id="legendEdges">%%LEGEND_EDGES%%</div>
      </div>
    </div>
  </aside>
  <main id="main">
    <section id="graph"><div id="cy"></div></section>
    <section id="detail"><div id="cy-detail"></div></section>
  </main>
  <section id="tables">
    <div style="padding:10px">
      <h3 style="margin:0 0 10px 0;" id="downTitle">%%DOWNSTREAM%%</h3>
      <div class="inline" style="margin-bottom:6px"><button id="btnExportOut">%%EXPORT_CSV_OUT%%</button></div>
      <table id="tblOut"><thead><tr><th>Target</th><th id="thIsReg">%%IS_REG%%</th></tr></thead><tbody></tbody></table>
    </div>
    <div style="padding:10px">
      <h3 style="margin:0 0 10px 0;" id="upTitle">%%UPSTREAM%%</h3>
      <div class="inline" style="margin-bottom:6px"><button id="btnExportIn">%%EXPORT_CSV_IN%%</button></div>
      <table id="tblIn"><thead><tr><th>Regulator</th></tr></thead><tbody></tbody></table>
    </div>
  </section>
  <footer id="footer">%%FOOTER%%</footer>
</div>

<!-- Cytoscape.js via CDN; for offline, replace with local file path -->
<script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
<script>
const DATA = %%DATA%%;
const I18N = %%I18N%%;
let LANG = 'zh';

// i18n helper
function t(key, params = {}){
  let s = (I18N[LANG] && I18N[LANG][key]) || (I18N['en'][key]) || key;
  Object.keys(params).forEach(k => { s = s.replace('{'+k+'}', params[k]); });
  return s;
}
function applyI18n(){
  document.title = t('title');
  document.getElementById('hdr-title').innerHTML = '<strong>' + t('title') + '</strong>';
  document.getElementById('hdr-sub').textContent = t('subtitle');
  document.getElementById('lbl-search').textContent = t('search_label');
  document.getElementById('btnShowAll').textContent = t('buttons_show_all');
  document.getElementById('btnEgo').textContent = t('buttons_ego');
  document.getElementById('lbl-tree').textContent = t('filters_layout');
  document.getElementById('btnOutTree').textContent = t('buttons_out_tree');
  document.getElementById('btnInTree').textContent = t('buttons_in_tree');
  document.getElementById('depthText').innerHTML = t('depth') + ': <b>' + document.getElementById('depthSlider').value + '</b>';
  document.getElementById('lbl-path').textContent = t('path_label');
  document.getElementById('pathSrc').placeholder = t('path_src');
  document.getElementById('pathDst').placeholder = t('path_dst');
  document.getElementById('btnPath').textContent = t('path_go');
  document.getElementById('lbl-filters').textContent = t('filters_layout');
  document.getElementById('lbl-edgefilters').textContent = t('edge_filters');
  document.getElementById('btnAllEdges').textContent = t('all_edges');
  document.getElementById('btnRegRegEdges').textContent = t('regreg_edges');
  document.getElementById('lbl-degfilter').childNodes[0].textContent = t('deg_filter');
  document.getElementById('lbl-layout').textContent = t('layout');
  document.getElementById('btnLayout').textContent = t('apply_layout');
  document.getElementById('btnPNG').textContent = t('export_png');
  document.getElementById('nodeInfoTitle').textContent = t('node_info');
  document.getElementById('legendNodes').textContent = t('legend_nodes');
  document.getElementById('legendEdges').textContent = t('legend_edges');
  document.getElementById('downTitle').innerHTML = t('downstream');
  document.getElementById('upTitle').innerHTML = t('upstream');
  document.getElementById('thIsReg').textContent = t('is_reg');
  document.getElementById('btnExportOut').textContent = t('export_csv_out');
  document.getElementById('btnExportIn').textContent = t('export_csv_in');
  document.getElementById('footer').textContent = t('footer');
}

// Styles for Cytoscape
const baseStyle = [
  { selector: 'node', style: { 'label': 'data(label)', 'font-size': 9, 'text-valign': 'center', 'text-halign': 'center', 'width': 16, 'height': 16, 'background-color': '#888', 'color': '#222' }},
  { selector: 'node[type = "regulator_only"]', style: {'background-color': '#6fa8dc'} },
  { selector: 'node[type = "target_only"]', style: {'background-color': '#f9cb9c'} },
  { selector: 'node[type = "both"]',         style: {'background-color': '#93c47d'} },
  { selector: 'edge', style: { 'curve-style': 'bezier', 'width': 1, 'line-color': '#999', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#999' }},
  { selector: 'edge[isRegToReg = "true"]', style: { 'line-style': 'solid' } },
  { selector: 'edge[isRegToReg = "false"]', style: { 'line-style': 'dashed', 'opacity': 0.7 } },
  { selector: '.faded', style: { 'opacity': 0.15 } },
  { selector: '.highlight', style: { 'background-color': '#ff6666', 'line-color': '#ff6666', 'target-arrow-color': '#ff6666' } }
];

const elements = DATA.elements;

// Populate datalist
const nodeList = document.getElementById('nodeList');
elements.nodes.forEach(n => { const o = document.createElement('option'); o.value = n.data.id; nodeList.appendChild(o); });

// Init Cytoscape
const cy = cytoscape({ container: document.getElementById('cy'), elements, style: baseStyle, layout: { name: 'cose' }, wheelSensitivity: 0.2 });
const cyDetail = cytoscape({ container: document.getElementById('cy-detail'), elements: [], style: baseStyle, layout: { name: 'concentric' }, wheelSensitivity: 0.2 });

function getNode(id){ return cy.$(`node[id = "${id}"]`); }
function getOut(id){ return DATA.outMap[id] || []; }
function getIn(id){ return DATA.inMap[id] || []; }

function updateNodeInfo(id){
  const n = cy.$(`node[id = "${id}"]`);
  if (!n || n.length === 0){ document.getElementById('nodeInfo').textContent = '—'; return; }
  const dt = n.data();
  const lines = [
    `${t('node_type')}: ${dt.type}`,
    `${t('node_outdeg')}: ${dt.out_to_reg}`,
    `${t('node_indeg')}: ${dt.in_from_reg}`,
    `${t('node_scc')}: ${dt.scc}`
  ];
  document.getElementById('nodeInfo').innerHTML = `<div><b>${dt.label}</b></div><div>${lines.join('<br/>')}</div>`;
}

function updateTables(id){
  const tbodyOut = document.querySelector('#tblOut tbody');
  const tbodyIn  = document.querySelector('#tblIn tbody');
  tbodyOut.innerHTML = ''; tbodyIn.innerHTML = '';

  const outs = (getOut(id) || []).slice().sort();
  outs.forEach(tg => {
    const isReg = DATA.regulators.includes(tg);
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${tg}</td><td>${isReg ? t('yes') : t('no')}</td>`;
    tbodyOut.appendChild(tr);
  });

  const ins = (getIn(id) || []).slice().sort();
  ins.forEach(rg => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${rg}</td>`;
    tbodyIn.appendChild(tr);
  });
}

function exportTableToCSV(tableId, filename){
  const rows = Array.from(document.querySelectorAll(`#${tableId} tr`));
  const csv = rows.map(r => Array.from(r.querySelectorAll('th,td')).map(td => `"${td.textContent.replace(/"/g,'""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function selectNode(id){
  const ele = getNode(id);
  if (!ele || ele.length === 0) return;
  cy.$('.highlight').removeClass('highlight');
  cy.$('.faded').removeClass('faded');
  const neigh = ele.closedNeighborhood();
  ele.addClass('highlight');
  cy.elements().difference(neigh).addClass('faded');
  cy.fit(neigh, 80);
  updateNodeInfo(id);
  updateTables(id);
  drawEgo(id);
}

function drawEgo(id){
  const center = getNode(id);
  const neigh = center.closedNeighborhood();
  cyDetail.elements().remove();
  cyDetail.add(neigh.jsons());
  cyDetail.layout({ name: 'concentric' }).run();
  cyDetail.fit(cyDetail.elements(), 50);
}

// BFS expand (out/in) up to depth
function bfsExpand(startId, direction='out', maxDepth=3){
  const visited = new Set([startId]);
  let frontier = [startId];
  const nodes = new Set([startId]);
  const edges = new Set();
  let depth = 0;
  while (frontier.length > 0 && depth < maxDepth){
    const next = [];
    for (const id of frontier){
      const list = (direction==='out') ? getOut(id) : getIn(id);
      for (const nb of list){
        const from = (direction==='out') ? id : nb;
        const to   = (direction==='out') ? nb : id;
        if (cy.$(`edge[source = "${from}"][target = "${to}"]`).length > 0){
          edges.add(from + '->' + to);
        }
        nodes.add(nb);
        if (!visited.has(nb)){ visited.add(nb); next.push(nb); }
      }
    }
    frontier = next;
    depth += 1;
  }
  return { nodes: Array.from(nodes), edges: Array.from(edges) };
}

function showTree(startId, kind){
  const depth = parseInt(document.getElementById('depthSlider').value || '3');
  const out = (kind==='out' || kind==='both') ? bfsExpand(startId, 'out', depth) : {nodes:[],edges:[]};
  const inn = (kind==='in'  || kind==='both') ? bfsExpand(startId, 'in',  depth) : {nodes:[],edges:[]};
  const nodeSet = new Set([startId, ...out.nodes, ...inn.nodes]);
  const edgeSet = new Set([...(out.edges||[]), ...(inn.edges||[])]);

  const eles = [];
  nodeSet.forEach(n => { const ele = cy.$(`node[id = "${n}"]`); if (ele.length > 0) eles.push(ele); });
  edgeSet.forEach(eid => { const [s,t] = eid.split('->'); const ele = cy.$(`edge[source = "${s}"][target = "${t}"]`); if (ele.length > 0) eles.push(ele); });

  cy.$('.faded').removeClass('faded');
  cy.$('.highlight').removeClass('highlight');
  const sub = cy.collection(eles);
  sub.addClass('highlight');
  cy.elements().difference(sub).addClass('faded');

  // mirror subgraph to detail panel
  cyDetail.elements().remove();
  cyDetail.add(sub.jsons());
  cyDetail.layout({ name: 'breadthfirst', directed: true, spacingFactor: 1.2 }).run();
  cyDetail.fit(cyDetail.elements(), 50);
}

function findShortestPath(src, dst){
  if (src === dst) return [src];
  const q = [src];
  const prev = new Map(); prev.set(src, null);
  while (q.length){
    const cur = q.shift();
    for (const nb of getOut(cur)){
      if (!prev.has(nb)){ prev.set(nb, cur); q.push(nb); }
      if (nb === dst){ q.length = 0; break; }
    }
  }
  if (!prev.has(dst)) return null;
  const path = [];
  let cur = dst;
  while (cur !== null){ path.push(cur); cur = prev.get(cur); }
  path.reverse(); return path;
}

function highlightPath(path){
  cy.$('.faded').removeClass('faded');
  cy.$('.highlight').removeClass('highlight');
  const eles = [];
  for (let i=0; i<path.length; i++){
    const n = cy.$(`node[id = "${path[i]}"]`);
    if (n.length > 0) eles.push(n);
    if (i < path.length-1){
      const e = cy.$(`edge[source = "${path[i]}"][target = "${path[i+1]}"]`);
      if (e.length > 0) eles.push(e);
    }
  }
  const sub = cy.collection(eles);
  sub.addClass('highlight');
  cy.elements().difference(sub).addClass('faded');
  cy.fit(sub, 80);

  cyDetail.elements().remove();
  cyDetail.add(sub.jsons());
  cyDetail.layout({ name: 'breadthfirst', directed: true, spacingFactor: 1.2 }).run();
  cyDetail.fit(cyDetail.elements(), 50);
}

// Filters
let showRegRegOnly = false;
function applyEdgeFilter(){
  cy.$('edge').removeClass('faded');
  if (showRegRegOnly){
    cy.$('edge[isRegToReg = "false"]').addClass('faded');
  }
}

let degThreshold = 0;
function applyDegreeFilter(){
  cy.$('node').removeClass('faded');
  if (degThreshold > 0){
    cy.nodes().forEach(n => {
      const od = n.data('out_to_reg') || 0;
      if (od < degThreshold) n.addClass('faded');
    });
  }
}

// Layout
function applyLayout(name){ cy.layout({ name }).run(); }

// Wire UI
document.getElementById('btnShowAll').onclick = () => { cy.$('.faded').removeClass('faded'); cy.$('.highlight').removeClass('highlight'); cy.fit(cy.elements(), 40); };
document.getElementById('btnEgo').onclick = () => { const id = document.getElementById('nodeSearch').value.trim(); if (id) { selectNode(id); } };
document.getElementById('btnOutTree').onclick = () => { const id = document.getElementById('nodeSearch').value.trim(); if (id) showTree(id, 'out'); };
document.getElementById('btnInTree').onclick = () => { const id = document.getElementById('nodeSearch').value.trim(); if (id) showTree(id, 'in'); };

document.getElementById('depthSlider').oninput = (e) => {
  document.getElementById('depthText').innerHTML = t('depth') + ': <b>' + e.target.value + '</b>';
};

document.getElementById('btnPath').onclick = () => {
  const s = document.getElementById('pathSrc').value.trim();
  const d = document.getElementById('pathDst').value.trim();
  if (s && d){
    const p = findShortestPath(s,d);
    if (p) highlightPath(p); else alert(t('not_found_path', {s, d}));
  }
};

document.getElementById('btnAllEdges').onclick = () => { showRegRegOnly = false; applyEdgeFilter(); applyDegreeFilter(); };
document.getElementById('btnRegRegEdges').onclick = () => { showRegRegOnly = true; applyEdgeFilter(); applyDegreeFilter(); };

const slider = document.getElementById('degSlider');
const degVal = document.getElementById('degVal');
slider.oninput = (e) => { degThreshold = parseInt(e.target.value); degVal.textContent = degThreshold; applyDegreeFilter(); };

document.getElementById('btnLayout').onclick = () => { const name = document.getElementById('layoutSel').value; applyLayout(name); };

document.getElementById('btnPNG').onclick = () => {
  const png64 = cy.png({ full: false, scale: 2, bg: '#ffffff' });
  const a = document.createElement('a');
  a.href = png64; a.download = 'regnet_view.png'; a.click();
};

document.getElementById('btnExportOut').onclick = () => { exportTableToCSV('tblOut', 'downstream.csv'); };
document.getElementById('btnExportIn').onclick = () => { exportTableToCSV('tblIn',  'upstream.csv'); };

cy.on('tap', 'node', (evt) => {
  const id = evt.target.id();
  document.getElementById('nodeSearch').value = id;
  selectNode(id);
});

document.getElementById('langSel').onchange = (e) => {
  LANG = e.target.value || 'zh';
  applyI18n();
};

// Initialize
applyI18n();
cy.fit(cy.elements(), 60);
</script>
</body>
</html>
"""

    html = (template
            .replace("%%TITLE%%", i18n["zh"]["title"])
            .replace("%%SUBTITLE%%", i18n["zh"]["subtitle"])
            .replace("%%SEARCH_LABEL%%", i18n["zh"]["search_label"])
            .replace("%%BTN_SHOWALL%%", i18n["zh"]["buttons_show_all"])
            .replace("%%BTN_EGO%%", i18n["zh"]["buttons_ego"])
            .replace("%%BTN_OUTTREE%%", i18n["zh"]["buttons_out_tree"])
            .replace("%%BTN_INTREE%%", i18n["zh"]["buttons_in_tree"])
            .replace("%%DEPTH%%", i18n["zh"]["depth"])
            .replace("%%PATH_LABEL%%", i18n["zh"]["path_label"])
            .replace("%%PATH_SRC%%", i18n["zh"]["path_src"])
            .replace("%%PATH_DST%%", i18n["zh"]["path_dst"])
            .replace("%%PATH_GO%%", i18n["zh"]["path_go"])
            .replace("%%FILTERS_LAYOUT%%", i18n["zh"]["filters_layout"])
            .replace("%%EDGE_FILTERS%%", i18n["zh"]["edge_filters"])
            .replace("%%ALL_EDGES%%", i18n["zh"]["all_edges"])
            .replace("%%REGREG_EDGES%%", i18n["zh"]["regreg_edges"])
            .replace("%%DEG_FILTER%%", i18n["zh"]["deg_filter"])
            .replace("%%LAYOUT%%", i18n["zh"]["layout"])
            .replace("%%APPLY_LAYOUT%%", i18n["zh"]["apply_layout"])
            .replace("%%EXPORT_PNG%%", i18n["zh"]["export_png"])
            .replace("%%NODE_INFO%%", i18n["zh"]["node_info"])
            .replace("%%DOWNSTREAM%%", i18n["zh"]["downstream"])
            .replace("%%UPSTREAM%%", i18n["zh"]["upstream"])
            .replace("%%IS_REG%%", i18n["zh"]["is_reg"])
            .replace("%%EXPORT_CSV_OUT%%", i18n["zh"]["export_csv_out"])
            .replace("%%EXPORT_CSV_IN%%", i18n["zh"]["export_csv_in"])
            .replace("%%LEGEND_NODES%%", i18n["zh"]["legend_nodes"])
            .replace("%%LEGEND_EDGES%%", i18n["zh"]["legend_edges"])
            .replace("%%FOOTER%%", i18n["zh"]["footer"])
            .replace("%%DATA%%", json.dumps({
                "elements": data["elements"],
                "outMap": data["outMap"],
                "inMap": data["inMap"],
                "regulators": data["regulators"]
            }))
            .replace("%%I18N%%", json.dumps(i18n))
            )

    output_path.write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to regulator→target mapping table (TSV/CSV/any delims)")
    p.add_argument("--output", default="RegNet_Explorer_Bilingual.html", help="Output HTML filename")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    df = load_mapping(inp)
    build_html(df, Path(args.output))
    print(f"Done. Open in browser: {args.output}")


if __name__ == "__main__":
    main()