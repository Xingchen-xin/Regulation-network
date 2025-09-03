"""
Minimal MCP tool server exposing RegNet functions to agents.
Requires: pip install mcp
Docs: https://modelcontextprotocol.io
"""
from pathlib import Path
from typing import List, Optional
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool

from backend.data_loader import DataStore

app = FastMCP("regnet-tools")
DATA = DataStore()

@app.on_startup()
async def _startup(_ctx=None):
    DATA.load_from_config(Path(__file__).resolve().parents[1] / "config.yaml")

@app.tool()
def list_nodes(prefix: str = "SCO", limit: int = 1000) -> List[str]:
    "List node IDs with a given prefix."
    return [g for g in sorted(DATA.nodes) if g.startswith(prefix)][:limit]

@app.tool()
def get_annotation(gene_id: str) -> dict:
    "Get annotation for a gene."
    return DATA.get_annotation(gene_id)

@app.tool()
def get_regulators(gene_id: str) -> List[str]:
    "List regulators of a gene."
    gid = gene_id.strip()
    return sorted(list(DATA.target_to_regs.get(gid, set())))

@app.tool()
def get_targets(gene_id: str) -> List[str]:
    "List targets of a regulator."
    gid = gene_id.strip()
    return sorted(list(DATA.reg_to_targets.get(gid, set())))

@app.tool()
def neighborhood(gene_id: str, hops: int = 1, direction: str = "both",
                 rr_only: bool = False, tf_min: int = 0, tgt_min: int = 0) -> dict:
    "Return neighborhood subgraph nodes/edges with filters."
    return DATA.neighborhood(gene_id, hops=hops, direction=direction, rr_only=rr_only, tf_min=tf_min, tgt_min=tgt_min)

@app.tool()
def shortest_path(source: str, target: str) -> dict:
    "Return nodes/edges along shortest directed path."
    return DATA.shortest_path(source, target)

@app.tool()
def enrich(genes: List[str], field: str, min_count: int = 2) -> List[dict]:
    "Over-representation analysis for a list of genes using an annotation field (GO/KEGG/PFAM)."
    return DATA.enrich(genes, field, min_count=min_count, universe=None)

if __name__ == "__main__":
    app.run()