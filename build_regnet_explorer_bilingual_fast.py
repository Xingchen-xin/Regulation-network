# save as: build_regnet_explorer_bilingual_fast.py
# usage:
#   pip install pandas networkx
#   python build_regnet_explorer_bilingual_fast.py \
#       --input Streptomyces_coelicolor_A32_regulator_to_target_analysis.tsv.txt \
#       --output RegNet_Explorer_Bilingual_FAST.html
import argparse
import json
import re
from pathlib import Path
import pandas as pd
import networkx as nx


def load_mapping(path: Path) -> pd.DataFrame:
    encs = ["utf-8", "utf-8-sig", "latin1"]
    seps = ["\t", ",", ";", "|"]
    for enc in encs:
        for sep in seps:
            try:
                tmp = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                cols = [c.lower() for c in tmp.columns]
                if any("reg" in c for c in cols) and any("target" in c for c in cols):
                    df = tmp.copy()
                    # normalize headers
                    colmap = {}
                    for c in df.columns:
                        cl = c.lower()
                        if "reg" in cl and "regulator" not in colmap.values():
                            colmap[c] = "regulator"
                        elif "target" in cl and "target" not in colmap.values():
                            colmap[c] = "target"
                    df = df.rename(columns=colmap)
                    df["target"] = df["target"].astype(str)
                    if df["target"].str.contains(r"[;, ]").any():
                        df = df.assign(target=df["target"].str.split(
                            r"[;,\s]+")).explode("target")
                    return df[["regulator", "target"]]
            except Exception:
                pass
    # fallback
    regs, tars = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
            for t in re.split(r"[,\s;]+", rest.strip()):
                if t:
                    regs.append(reg.strip())
                    tars.append(t.strip())
    return pd.DataFrame({"regulator": regs, "target": tars})


def norm_sco(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x)
    # 去掉不可见空格与常见空白
    s = s.replace("\xa0", "").strip()
    # 标准化大小写
    s = s.replace("sco", "SCO").replace("SCo", "SCO").replace("ScO", "SCO")
    # 去除中间残留空白（如 "SCO 4434"）
    s = re.sub(r"\s+", "", s)
    return s

def build(df: pd.DataFrame, out_html: Path):
    df = df.copy()
    df["regulator"] = df["regulator"].map(norm_sco)
    df["target"] = df["target"].map(norm_sco)
    df = df.dropna(subset=["regulator", "target"])
    df = df[(df["regulator"] != "") & (df["target"] != "")]
    edges_all = df[["regulator", "target"]].drop_duplicates()
    regs = set(edges_all["regulator"].unique())
    targets = set(edges_all["target"].unique())

    # Reg→Reg subgraph
    edges_rr = edges_all[edges_all["target"].isin(regs)]
    edges_rr = edges_rr[edges_rr["regulator"] != edges_rr["target"]]
    Grr = nx.from_pandas_edgelist(
        edges_rr, source="regulator", target="target", create_using=nx.DiGraph())

    out_deg = dict(Grr.out_degree())
    in_deg = dict(Grr.in_degree())
    # spring_layout 预计算坐标（在 Reg→Reg 上更稳），再把仅 target 的节点周边放置
    pos = nx.spring_layout(Grr, seed=42, dim=2, k=None, iterations=200)
    # 给仅作为 target 的节点一个近似位置：放到其任一上游的附近
    from collections import defaultdict
    out_map = defaultdict(list)
    in_map = defaultdict(list)
    for r, t in edges_all.itertuples(index=False):
        out_map[r].append(t)
        in_map[t].append(r)
    nodes_all = regs.union(targets)
    for n in nodes_all:
        if n not in pos:
            ups = in_map.get(n, [])
            assigned = False
            for base in ups:
                if base in pos:
                    pos[n] = (pos[base][0] + 0.02, pos[base][1] + 0.02)
                    assigned = True
                    break
            if not assigned:
                downs = out_map.get(n, [])
                for nb in downs:
                    if nb in pos:
                        pos[n] = (pos[nb][0] - 0.02, pos[nb][1] - 0.02)
                        assigned = True
                        break
            if not assigned:
                pos[n] = (0.0, 0.0)
    # nodes payload with preset positions
    nodes_set = regs.union(targets)
    node_records = []
    for n in nodes_set:
        if n in regs and n in targets:
            ntype = "both"
        elif n in regs:
            ntype = "regulator_only"
        else:
            ntype = "target_only"
        _x, _y = pos.get(n, (0.0, 0.0))
        node_records.append({
            "data": {
                "id": n, "label": n, "type": ntype,
                "out_to_reg": int(out_deg.get(n, 0)),
                "in_from_reg": int(in_deg.get(n, 0))
            },
            "position": {"x": float(_x)*1000, "y": float(_y)*1000}
        })

    # edges payload
    edge_records = []
    for r, t in edges_all.itertuples(index=False):
        edge_records.append({
            "data": {"id": f"{r}->{t}", "source": r, "target": t, "isRegToReg": bool(t in regs)}
        })

    data = {
        "elements": {"nodes": node_records, "edges": edge_records},
        "outMap": out_map, "inMap": in_map, "regulators": sorted(list(regs))
    }

    # 初始“精简子网”：Top 出度≥10 的枢纽及其 1 跳
    hubs = [n for n, deg in out_deg.items() if deg >= 10]
    slim_nodes = set(hubs)
    for h in hubs:
        slim_nodes.update(out_map.get(h, []))
        for up in in_map.get(h, []):
            slim_nodes.add(up)
    slim_edges = [e for e in edge_records if (
        e["data"]["source"] in slim_nodes and e["data"]["target"] in slim_nodes)]

    i18n = {
        "zh": {"title": "RegNet Explorer — FAST", "subtitle": "预计算布局 / 首屏精简 / 按需加载全网",
               "search": "选择/搜索节点", "show_all": "载入全网", "ego": "显示1跳邻域", "out": "向外树", "in": "向内树",
               "depth": "树深度", "path": "最短路径", "src": "源节点", "dst": "目标节点", "go": "查找",
               "filters": "过滤与布局", "all": "全部边", "regreg": "只看Reg→Reg", "degree": "对regulator出度≥ ",
               "layout": "布局", "apply": "应用", "png": "导出PNG", "info": "节点信息",
               "down": "下游（targets）", "up": "上游（regulators）", "isreg": "是否为Regulator",
               "legend1": "节点：regulator_only / target_only / both", "legend2": "边：Reg→Reg 实线；Reg→Target 虚线",
               "tip": "提示：该版本默认隐藏标签、无箭头、直线边；放大/选中时再显示细节以保证流畅。"},
        "en": {"title": "RegNet Explorer — FAST", "subtitle": "Preset layout / slim first-view / on-demand full load",
               "search": "Select/Search Node", "show_all": "Load Full Network", "ego": "Show 1-hop", "out": "Out-tree", "in": "In-tree",
               "depth": "Tree Depth", "path": "Shortest Path", "src": "Source", "dst": "Target", "go": "Find",
               "filters": "Filters & Layout", "all": "All edges", "regreg": "Reg→Reg only", "degree": "Out-degree to regs ≥ ",
               "layout": "Layout", "apply": "Apply", "png": "Export PNG", "info": "Node Info",
               "down": "Downstream (targets)", "up": "Upstream (regulators)", "isreg": "Is Regulator",
               "legend1": "Nodes: regulator_only / target_only / both", "legend2": "Edges: Reg→Reg solid; Reg→Target dashed",
               "tip": "Note: Labels hidden, no arrows, straight edges by default; details appear on zoom/select for performance."}
    }

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>RegNet Explorer — FAST</title><meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
body{{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}}
#app{{display:grid;grid-template-columns:340px 1fr;grid-template-rows:auto 1fr auto;height:100vh}}
header{{grid-column:1/3;display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-bottom:1px solid #eee;background:#fafafa}}
#sidebar{{padding:10px;border-right:1px solid #eee;overflow:auto}}
#main{{position:relative}} #cy{{position:absolute;inset:0}}
#detail{{height:35vh;border-top:1px solid #eee;position:relative}} #cy2{{position:absolute;inset:0}}
#tables{{display:grid;grid-template-columns:1fr 1fr;max-height:40vh;overflow:auto}}
table{{width:100%;border-collapse:collapse;font-size:12px}} th,td{{border-bottom:1px solid #eee;padding:6px 8px;text-align:left}}
button{{padding:6px 10px;border:1px solid #ddd;background:#fff;border-radius:8px;cursor:pointer;margin-right:6px}}
.tiny{{font-size:12px;color:#666}}
</style>
</head><body>
<div id="app">
<header><div><b id="t-title">FAST RegNet</b><div class="tiny" id="t-sub"></div></div>
<div><select id="lang"><option value="zh">中文</option><option value="en">English</option></select></div></header>
<aside id="sidebar">
  <div class="tiny" id="t-search"></div>
  <input id="q" list="nodes" placeholder="SCO0110 / SCO5261 / redD"/>
  <datalist id="nodes"></datalist>
  <div style="margin:8px 0">
    <button id="btnSlim">Slim View</button>
    <button id="btnFull"></button>
    <button id="btnEgo"></button>
  </div>
  <div style="margin:8px 0" class="tiny" id="t-filters"></div>
  <div>
    <button id="btnAll"></button>
    <button id="btnRegReg"></button>
  </div>
  <div class="tiny" style="margin-top:6px"><span id="t-degree"></span><b id="degv">0</b></div>
  <input type="range" id="deg" min="0" max="80" step="1" value="0"/>
  <div class="tiny" style="margin-top:6px"><span id="t-layout"></span>
    <select id="layout"><option>preset</option><option>circle</option><option>breadthfirst</option></select>
    <button id="apply"></button>
  </div>
  <div style="margin-top:6px"><button id="png"></button></div>
  <h4 id="t-info" style="margin:8px 0"></h4>
  <div id="info" class="tiny">—</div>
  <div class="tiny" style="margin-top:8px" id="t-legend1"></div>
  <div class="tiny" id="t-legend2"></div>
  <div class="tiny" style="margin-top:8px" id="t-tip"></div>
</aside>
<main id="main"><div id="cy"></div></main>
<section id="detail"><div id="cy2"></div></section>
<section id="tables">
  <div style="padding:8px"><h3 id="t-down"></h3><table id="down"><thead><tr><th>Target</th><th id="t-isreg"></th></tr></thead><tbody></tbody></table></div>
  <div style="padding:8px"><h3 id="t-up"></h3><table id="up"><thead><tr><th>Regulator</th></tr></thead><tbody></tbody></table></div>
</section>
</div>
<script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>
<script>
const I18N={json.dumps(i18n)};
let LANG='zh';
const FULL={json.dumps(data)};    // 全量
const SLIM={{elements:{{nodes:{json.dumps([n for n in node_records if n["data"]["id"] in slim_nodes])},edges:{json.dumps(slim_edges)}}}}}; // 首屏精简
function t(k){{return (I18N[LANG]&&I18N[LANG][k])||I18N.en[k]||k}}
function text(){{
  document.getElementById('t-title').textContent=t('title');
  document.getElementById('t-sub').textContent=t('subtitle');
  document.getElementById('t-search').textContent=t('search');
  document.getElementById('btnFull').textContent=t('show_all');
  document.getElementById('btnEgo').textContent=t('ego');
  document.getElementById('t-filters').textContent=t('filters');
  document.getElementById('btnAll').textContent=t('all');
  document.getElementById('btnRegReg').textContent=t('regreg');
  document.getElementById('t-degree').textContent=t('degree');
  document.getElementById('t-layout').textContent=t('layout');
  document.getElementById('apply').textContent=t('apply');
  document.getElementById('png').textContent=t('png');
  document.getElementById('t-info').textContent=t('info');
  document.getElementById('t-legend1').textContent=t('legend1');
  document.getElementById('t-legend2').textContent=t('legend2');
  document.getElementById('t-tip').textContent=t('tip');
  document.getElementById('t-down').textContent=t('down');
  document.getElementById('t-up').textContent=t('up');
  document.getElementById('t-isreg').textContent=t('isreg');
}}
document.getElementById('lang').onchange=e=>{{LANG=e.target.value;text()}};
text();

// 节点列表
const dl=document.getElementById('nodes');
FULL.elements.nodes.forEach(n=>{{const o=document.createElement('option');o.value=n.data.id;dl.appendChild(o)}});

// 性能友好样式：直线边/无箭头/隐藏标签（高亮时再开）
const baseStyle=[
 {{ selector:'core', style:{{'selection-box-color':'#ddd','selection-box-border-color':'#aaa'}} }},
 {{ selector:'node', style:{{'width':14,'height':14,'label':'','font-size':9,'text-valign':'center','text-halign':'center','background-color':'#888'}} }},
 {{ selector:'node[type = "regulator_only"]', style:{{'background-color':'#6fa8dc'}} }},
 {{ selector:'node[type = "target_only"]', style:{{'background-color':'#f9cb9c'}} }},
 {{ selector:'node[type = "both"]', style:{{'background-color':'#93c47d'}} }},
 {{ selector:'edge', style:{{'curve-style':'haystack','haystack-radius':2,'width':0.8,'line-color':'#bbb'}} }},
 {{ selector:'.showLabel', style:{{'label':'data(label)'}} }},
 {{ selector:'.hiEdge', style:{{'line-color':'#ff6666','width':2,'curve-style':'bezier','target-arrow-shape':'triangle','target-arrow-color':'#ff6666'}} }},
 {{ selector:'.hiNode', style:{{'background-color':'#ff6666'}} }},
 {{ selector:'.fade', style:{{'opacity':0.15}} }}
];

// 初始化（采用 preset，不跑浏览器布局；并开启渲染优化）
let cy=cytoscape({{
  container:document.getElementById('cy'),
  elements:SLIM.elements,
  style:baseStyle,
  layout:{{name:'preset'}},
  wheelSensitivity:0.2,
  pixelRatio:1,
  textureOnViewport:true,
  motionBlur:true,
  motionBlurOpacity:0.15,
  hideEdgesOnViewport:true,
  hideLabelsOnViewport:true
}});
let cy2=cytoscape({{container:document.getElementById('cy2'), elements:[], style:baseStyle, layout:{{name:'preset'}}, wheelSensitivity:0.2, pixelRatio:1 }});
cy.fit(cy.elements(), 40);

// 交互函数
function info(id){{
  const n=cy.$(`node[id = "${{id}}"]`); if(!n) return;
  const d=n.data();
  document.getElementById('info').innerHTML=`<div><b>${{d.label}}</b></div>
    <div>type: ${{d.type}}</div><div>out_to_reg: ${{d.out_to_reg}}</div><div>in_from_reg: ${{d.in_from_reg}}</div>`;
}}
function selectNode(id){{
  cy.startBatch();
  cy.$('.hiNode').removeClass('hiNode'); cy.$('.hiEdge').removeClass('hiEdge'); cy.$('.fade').removeClass('fade'); cy.$('.showLabel').removeClass('showLabel');
  const n=cy.$(`node[id = "${{id}}"]`); if(n.nonempty){{}}
  const neigh=n.closedNeighborhood();
  n.addClass('hiNode showLabel'); neigh.nodes().addClass('showLabel');
  cy.elements().difference(neigh).addClass('fade');
  cy.endBatch(); cy.fit(neigh, 80);
  info(id); updateTables(id); drawEgo(neigh);
}}
function drawEgo(eles){{
  cy2.elements().remove(); cy2.add(eles.jsons()); cy2.fit(cy2.elements(),60);
}}
function updateTables(id){{
  function td(s){{return `<td>${{s}}</td>`}}
  const down=document.querySelector('#down tbody'); const up=document.querySelector('#up tbody'); down.innerHTML=''; up.innerHTML='';
  const outs=(FULL.outMap[id]||[]).slice().sort(); const regs=new Set(FULL.regulators);
  outs.forEach(t=>{{down.insertAdjacentHTML('beforeend',`<tr>${{td(t)}}${{td(regs.has(t)?'Yes/是':'No/否')}}</tr>`)}});
  const ins=(FULL.inMap[id]||[]).slice().sort(); ins.forEach(r=>{{up.insertAdjacentHTML('beforeend',`<tr>${{td(r)}}</tr>`)}});
}}

// UI 绑定
document.getElementById('btnSlim').onclick=()=>{{ // 回到精简视图
  cy.json({{elements:SLIM.elements}}); cy.layout({{name:'preset'}}).run(); cy.fit(cy.elements(),40);
}};
document.getElementById('btnFull').onclick=()=>{{ // 按需加载全网
  cy.startBatch(); cy.elements().remove(); cy.add(FULL.elements); cy.endBatch(); cy.layout({{name:'preset'}}).run(); cy.fit(cy.elements(),40);
}};
document.getElementById('btnEgo').onclick=()=>{{ const id=document.getElementById('q').value.trim(); if(id) selectNode(id); }};
document.getElementById('btnAll').onclick=()=>{{ cy.$('.fade').removeClass('fade'); }};
document.getElementById('btnRegReg').onclick=function() {{
        cy.batch(function() {{ cy.$('edge[isRegToReg = "false"]').addClass('fade'); }});
    }};
document.getElementById('apply').onclick=()=>{{ const name=document.getElementById('layout').value; cy.layout({{name}}).run(); }};
document.getElementById('png').onclick=()=>{{ const a=document.createElement('a'); a.href=cy.png({{full:false,scale:2,bg:'#fff'}}); a.download='view.png'; a.click(); }};
document.getElementById('deg').oninput=(e)=>{{ const th=parseInt(e.target.value); document.getElementById('degv').textContent=th; cy.batch(()=>{{
  cy.$('node').removeClass('fade'); if(th>0) cy.nodes().forEach(n=>{{ if((n.data('out_to_reg')||0)<th) n.addClass('fade'); }});
}})}};
cy.on('tap','node',evt=>{{ document.getElementById('q').value=evt.target.id(); selectNode(evt.target.id()); }});
</script></body></html>"""
    out_html.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="RegNet_Explorer_Bilingual_FAST.html")
    args = ap.parse_args()
    df = load_mapping(Path(args.input))
    build(df, Path(args.output))
    print("OK:", args.output)


if __name__ == "__main__":
    main()
