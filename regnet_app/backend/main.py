from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import yaml

from .data_loader import DataStore

app = FastAPI(title="RegNet Annotated API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA = DataStore()

class ConfigIn(BaseModel):
    map: str
    anno: str
    id_col: Optional[str] = None

class EnrichIn(BaseModel):
    genes: List[str]
    field: str
    min_count: int = 2
    universe: Optional[List[str]] = None

class PathIn(BaseModel):
    source: str
    target: str

@app.on_event("startup")
def startup():
    DATA.load_from_config(Path(__file__).resolve().parents[1] / "config.yaml")

@app.get("/api/health")
def health():
    return {"status":"ok","nodes":len(DATA.nodes),"edges":int(DATA.rt.shape[0])}

@app.post("/api/reload")
def reload(cfg: Optional[ConfigIn] = None):
    if cfg:
        (Path(__file__).resolve().parents[1] / "config.yaml").write_text(
            yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8"
        )
    DATA.load_from_config(Path(__file__).resolve().parents[1] / "config.yaml")
    return {"status":"reloaded","nodes":len(DATA.nodes),"edges":int(DATA.rt.shape[0])}

@app.get("/api/meta")
def meta():
    return {
        "gene_col": DATA.gene_col,
        "columns": list(DATA.anno.columns),
        "regulators": len(DATA.reg_set),
        "targets": len(DATA.tgt_set),
        "nodes": len(DATA.nodes),
        "edges": int(DATA.rt.shape[0]),
    }

@app.get("/api/search")
def search(q: str, limit: int = 20):
    return {"items": DATA.search(q, limit)}

@app.get("/api/node/{gid}")
def node(gid: str):
    try:
        if gid not in DATA.nodes:
            raise HTTPException(404, f"{gid} not in network")
        payload = DATA.get_annotation(gid)
        return JSONResponse(content=jsonable_encoder(payload))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            500, f"node lookup failed: {type(e).__name__}: {e}")

@app.get("/api/neighborhood/{gid}")
def neighborhood(gid: str, hops: int = 1, direction: str = "both",
                 rr_only: bool = False, tf_min: int = 0, tgt_min: int = 0):
    if gid not in DATA.nodes:
        raise HTTPException(404, f"{gid} not in network")
    return DATA.neighborhood(gid, hops=hops, direction=direction, rr_only=rr_only, tf_min=tf_min, tgt_min=tgt_min)

@app.post("/api/path")
def path(body: PathIn):
    return DATA.shortest_path(body.source, body.target)

@app.get("/api/exports/hubs")
def export_hubs():
    return DATA.export_hubs().to_dict(orient="records")

@app.get("/api/exports/highly_regulated")
def export_highly_regulated():
    return DATA.export_highly_regulated().to_dict(orient="records")

@app.get("/api/exports/self_regulators")
def export_self_regulators():
    return DATA.export_self_regulators().to_dict(orient="records")

@app.get("/api/exports/scc")
def export_scc():
    return DATA.export_scc().to_dict(orient="records")

@app.post("/api/enrich")
def enrich(body: EnrichIn):
    res = DATA.enrich(body.genes, body.field, min_count=body.min_count, universe=body.universe)
    return {"items": res}


@app.get("/api/analysis/metrics")
def analysis_metrics():
    # 返回汇总指标（注意：中心性字典可能很大，这里只给前端常用的摘要）
    M = DATA.metrics or {}
    summary = {k: M.get(k) for k in [
        "number_of_nodes", "number_of_edges",
        "average_in_degree", "average_out_degree",
        "average_clustering_coefficient",
        "number_of_strongly_connected_components",
        "largest_strongly_connected_component",
        "topological_sort_sample",
    ]}
    return summary


@app.get("/api/analysis/key_nodes")
def analysis_key_nodes():
    if not DATA.key_nodes:
        DATA.compute_metrics()
    # SCC 转成 list[list] 便于 JSON
    out = dict(DATA.key_nodes)
    out["scc_top"] = [sorted(list(s))
                      for s in DATA.key_nodes.get("scc_top", [])]
    return out


@app.get("/api/analysis/two_step_paths")
def analysis_two_step_paths(top_k: int = 20):
    items = DATA.common_two_step_paths(top_k=top_k)
    return {"items": [{"path": p, "count": c} for p, c in items]}


@app.get("/api/exports/edges_full")
def export_edges_full():
    return DATA.export_edges().to_dict(orient="records")


@app.get("/api/exports/nodes_with_metrics")
def export_nodes_with_metrics():
    return DATA.export_nodes_with_metrics().to_dict(orient="records")


@app.get("/api/report")
def report():
    """极简 Markdown 报告，前端可直接下载为 .md"""
    M = DATA.metrics or {}
    lines = []
    lines.append("# Streptomyces 调控网络分析报告\n")
    lines.append(f"- 节点数: {M.get('number_of_nodes', '-')}")
    lines.append(f"- 边数: {M.get('number_of_edges', '-')}")
    lines.append(f"- 平均入度: {M.get('average_in_degree', '-')}")
    lines.append(f"- 平均出度: {M.get('average_out_degree', '-')}")
    lines.append(f"- 平均聚类系数: {M.get('average_clustering_coefficient', '-')}")
    lines.append(
        f"- 强连通分量数: {M.get('number_of_strongly_connected_components', '-')}")
    lines.append(
        f"- 最大强连通分量大小: {M.get('largest_strongly_connected_component', '-')}\n")
    return {"markdown": "\n".join(lines)}


@app.get("/api/motifs/ffl")
def api_motifs_ffl(rr_only: bool = False, limit: int = 50000):
    """返回 FFL (A->B->C 且 A->C)"""
    try:
        items = DATA.motifs_ffl(rr_only=rr_only, limit=limit)
        return {"items": items}
    except Exception as e:
        raise HTTPException(500, f"FFL scan failed: {e}")


@app.get("/api/motifs/bifan")
def api_motifs_bifan(min_shared: int = 2, rr_only: bool = True, limit: int = 50000, expand_limit_per_pair: int = 200):
    """返回 Bi-fan 四元组 (A,B,C,D)，满足 A,B 共同调控至少 min_shared 个下游，展开至限制"""
    try:
        items = DATA.motifs_bifan(min_shared=min_shared, rr_only=rr_only,
                                  limit=limit, expand_limit_per_pair=expand_limit_per_pair)
        return {"items": items}
    except Exception as e:
        raise HTTPException(500, f"Bi-fan scan failed: {e}")
