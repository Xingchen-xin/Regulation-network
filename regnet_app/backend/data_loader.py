from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
import yaml

def norm_sco(x):
    if pd.isna(x):
        return x
    s = str(x).replace("\xa0", "").strip()
    s = s.replace("sco","SCO").replace("SCo","SCO").replace("ScO","SCO")
    s = re.sub(r"\s+","", s)
    return s

def load_mapping(path: Path) -> pd.DataFrame:
    encs = ["utf-8","utf-8-sig","latin1"]
    seps = ["\t", ",", ";", "|"]
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                cols = [c.lower() for c in df.columns]
                if any("reg" in c for c in cols) and any("target" in c for c in cols):
                    colmap = {}
                    for c in df.columns:
                        cl = c.lower()
                        if "reg" in cl and "regulator" not in colmap.values():
                            colmap[c] = "regulator"
                        elif "target" in cl and "target" not in colmap.values():
                            colmap[c] = "target"
                    df = df.rename(columns=colmap)
                    df["regulator"] = df["regulator"].map(norm_sco)
                    df["target"] = df["target"].astype(str)
                    df = df.assign(target=df["target"].str.split(r"[;,\s]+")).explode("target")
                    df["target"] = df["target"].map(norm_sco)
                    df = df.dropna(subset=["regulator","target"])
                    df = df[(df["regulator"]!="") & (df["target"]!="")]
                    return df[["regulator","target"]].drop_duplicates()
            except Exception:
                pass
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
            for t in re.split(r"[;,\s]+", rest.strip()):
                if t:
                    regs.append(norm_sco(reg))
                    tars.append(norm_sco(t))
    return pd.DataFrame({"regulator": regs, "target": tars}).drop_duplicates()

def detect_gene_col(df: pd.DataFrame) -> Optional[str]:
    best, score = None, -1
    for c in df.columns:
        cl = c.strip().lower()
        sc = 0
        if cl in ["sco","sco_id","gene","geneid","gene_id","locus","locus_tag","scoid","sco_no","sco_number"]:
            sc += 10
        if "sco" in cl:
            sc += 5
        try:
            hit = df[c].astype(str).fillna("").str.contains(r"^S?CO\d{3,5}$", case=False).mean()
            sc += hit*10
        except Exception:
            pass
        if sc > score:
            best, score = c, sc
    return best

def parse_multivalue_cell(val: Any) -> List[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)): return []
    s = str(val).strip()
    if not s: return []
    return [p for p in re.split(r"[;,/|\s]+", s) if p]

def benjamini_hochberg(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m+1)
    pvals = np.array(pvals, dtype=float)
    q = pvals * m / ranks
    q[order[::-1]] = np.minimum.accumulate(q[order[::-1]])
    q = np.clip(q, 0, 1)
    return q.tolist()

def _to_jsonable(v: Any):
    import numpy as _np
    import pandas as _pd
    if v is None: return None
    if isinstance(v, float) and (_np.isnan(v) if isinstance(v, _np.floating) or isinstance(v, float) else False):
        return None
    if isinstance(v, (_np.floating, _np.integer)):
        return v.item()
    if isinstance(v, (_pd.Timestamp,)):
        return v.isoformat()
    return v

class DataStore:
    def __init__(self):
        self.map_path: Optional[Path] = None
        self.anno_path: Optional[Path] = None
        self.id_col: Optional[str] = None

        self.rt: Optional[pd.DataFrame] = None
        self.anno: Optional[pd.DataFrame] = None
        self.gene_col: Optional[str] = None

        self.reg_set: Set[str] = set()
        self.tgt_set: Set[str] = set()
        self.nodes: Set[str] = set()

        self.reg_to_targets: Dict[str, Set[str]] = {}
        self.target_to_regs: Dict[str, Set[str]] = {}

        self.reg2reg_graph: Optional[nx.DiGraph] = None
        self.scc_map: Dict[str, int] = {}

            # 在类属性里再加一个全图（包含所有边）
        self.G: Optional[nx.DiGraph] = None
        self.metrics: Dict[str, Any] = {}
        self.key_nodes: Dict[str, Any] = {}


    def load_from_config(self, config: Path):
        cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        self.map_path = Path(cfg["map"]).expanduser().resolve()
        self.anno_path = Path(cfg["anno"]).expanduser().resolve()
        self.id_col = cfg.get("id_col") or None

        self.rt = load_mapping(self.map_path)
        self.reg_set = set(self.rt["regulator"].unique())
        self.tgt_set = set(self.rt["target"].unique())
        self.nodes = self.reg_set | self.tgt_set

        reg2t = defaultdict(set); t2reg = defaultdict(set)
        for r, t in self.rt.itertuples(index=False):
            reg2t[r].add(t); t2reg[t].add(r)
        self.reg_to_targets, self.target_to_regs = dict(reg2t), dict(t2reg)

        self.anno = pd.read_csv(self.anno_path, encoding="utf-8")
        self.gene_col = self.id_col or detect_gene_col(self.anno)
        if self.gene_col is None:
            raise RuntimeError("Cannot detect gene id column in annotation; set id_col in config.yaml")

        # 统一规范化 ID，并显式创建 GeneID 列（前端就一定能看到）
        self.anno[self.gene_col] = self.anno[self.gene_col].map(norm_sco)
        self.anno["_ID_"] = self.anno[self.gene_col]
        self.anno["GeneID"] = self.anno["_ID_"]    # 关键：显式提供 GeneID 列

        self.anno["Regulating_TFs"]  = self.anno["_ID_"].map(lambda g: ",".join(sorted(self.target_to_regs.get(g, set()))))
        self.anno["TF_Count"]        = self.anno["_ID_"].map(lambda g: len(self.target_to_regs.get(g, set())))
        self.anno["Is_TF"]           = self.anno["_ID_"].map(lambda g: g in self.reg_set)
        self.anno["Regulates_Genes"] = self.anno["_ID_"].map(lambda g: ",".join(sorted(self.reg_to_targets.get(g, set()))))
        self.anno["Target_Count"]    = self.anno["_ID_"].map(lambda g: len(self.reg_to_targets.get(g, set())))

        rr_edges = self.rt[self.rt["target"].isin(self.reg_set)]
        self.reg2reg_graph = nx.from_pandas_edgelist(rr_edges, source="regulator", target="target", create_using=nx.DiGraph())
        self.scc_map = {}
        for i, comp in enumerate(nx.strongly_connected_components(self.reg2reg_graph)):
            for n in comp:
                self.scc_map[n] = i
        self.build_full_graph()
        self.compute_metrics(approx=True)

    # --------- Queries ---------
    def search(self, q: str, limit: int = 20) -> List[str]:
        q = (q or "").strip().lower()
        ids = sorted([g for g in self.nodes if q in g.lower()])[:limit]
        return ids

    def get_annotation(self, gene_id: str) -> dict:
        gid = norm_sco(gene_id)
        base = {
            "GeneID": gid,
            "Is_TF": gid in self.reg_set,
            "TF_Count": int(len(self.target_to_regs.get(gid, set()))),
            "Target_Count": int(len(self.reg_to_targets.get(gid, set()))),
        }
        row = self.anno[self.anno["_ID_"] == gid]
        if row.empty:
            return base
        d = row.iloc[0].to_dict()
        d.pop("_ID_", None)
        # 逐项转换为 JSON 友好的类型
        d = {k: _to_jsonable(v) for k, v in d.items()}
        d.update(base)
        return d

    def neighborhood(self, gene_id: str, hops: int = 1, direction: str = "both",
                     rr_only: bool = False, tf_min: int = 0, tgt_min: int = 0):
        gid = norm_sco(gene_id)
        nodes = set([gid])
        frontier = [gid]
        for _ in range(max(1, hops)):
            nxt = []
            for x in frontier:
                outs = list(self.reg_to_targets.get(x, set())) if direction in ("both","out") else []
                ins  = list(self.target_to_regs.get(x, set())) if direction in ("both","in")  else []
                for y in outs+ins:
                    nodes.add(y)
                    nxt.append(y)
            frontier = nxt
        edges = []
        for r,t in self.rt.itertuples(index=False):
            if r in nodes and t in nodes:
                if rr_only and (t not in self.reg_set): continue
                edges.append((r,t))
        def pass_filters(n):
            ann = self.get_annotation(n)
            if ann.get("TF_Count",0) < tf_min: return False
            if ann.get("Target_Count",0) < tgt_min: return False
            return True
        nodes = [n for n in nodes if pass_filters(n)]
        edges = [(r,t) for (r,t) in edges if (r in nodes and t in nodes)]
        return {"nodes": sorted(list(nodes)), "edges": edges}

    def shortest_path(self, source: str, target: str):
        s = norm_sco(source); t = norm_sco(target)
        G = nx.from_pandas_edgelist(self.rt, source="regulator", target="target", create_using=nx.DiGraph())
        try:
            path = nx.shortest_path(G, s, t)
            edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            return {"nodes": path, "edges": edges}
        except Exception:
            return {"nodes": [], "edges": []}

    # --------- Exports ---------
    def export_hubs(self) -> pd.DataFrame:
        df = self.anno.assign(_ID_=self.anno[self.gene_col].map(norm_sco))
        out = df[["_ID_", self.gene_col, "Is_TF", "Target_Count", "Regulates_Genes"]].drop_duplicates("_ID_")
        out = out[out["Is_TF"]==True].sort_values("Target_Count", ascending=False)
        out = out.rename(columns={self.gene_col:"GeneID"}).drop(columns=["_ID_"])
        return out

    def export_highly_regulated(self) -> pd.DataFrame:
        df = self.anno.assign(_ID_=self.anno[self.gene_col].map(norm_sco))
        out = df[["_ID_", self.gene_col, "TF_Count", "Regulating_TFs"]].drop_duplicates("_ID_")
        out = out.sort_values("TF_Count", ascending=False).rename(columns={self.gene_col:"GeneID"}).drop(columns=["_ID_"])
        return out

    def export_self_regulators(self) -> pd.DataFrame:
        rt_set = set(map(tuple, self.rt[["regulator","target"]].values.tolist()))
        auto = sorted([r for r in self.reg_set if (r,r) in rt_set])
        return pd.DataFrame({"GeneID": auto})

    def export_scc(self) -> pd.DataFrame:
        rows = []
        for i, comp in enumerate(nx.strongly_connected_components(self.reg2reg_graph)):
            for n in comp:
                rows.append({"SCC_ID": i, "GeneID": n, "Size": len(comp)})
        return pd.DataFrame(rows).sort_values(["Size","SCC_ID"], ascending=[False, True])

    # --------- Enrichment ---------
    def _build_term_index(self, field: str) -> Dict[str, Set[str]]:
        field = field.strip()
        if field not in self.anno.columns:
            return {}
        term2genes = defaultdict(set)
        for _, row in self.anno.iterrows():
            gid = norm_sco(row[self.gene_col])
            vals = parse_multivalue_cell(row[field])
            for v in vals:
                term2genes[v].add(gid)
        return term2genes

    def enrich(self, gene_list: List[str], field: str, min_count: int = 2, universe: Optional[List[str]] = None):
        from scipy.stats import fisher_exact
        genes = [norm_sco(g) for g in gene_list if g]
        genes = [g for g in genes if g in self.nodes]
        if len(genes)==0:
            return []
        term2genes = self._build_term_index(field)
        if not term2genes:
            return []

        if universe:
            U = set(norm_sco(g) for g in universe)
        else:
            U = set()
            for gset in term2genes.values(): U |= gset

        hits = set(g for g in genes if g in U)
        N = len(U); K = len(hits)
        rows = []
        for term, gset in term2genes.items():
            overlap = len(hits & gset)
            if overlap < min_count: continue
            a = overlap
            b = K - overlap
            c = len(gset - hits)
            d = N - a - b - c
            _, p = fisher_exact([[a,b],[c,d]], alternative="greater")
            rows.append((term, overlap, len(gset), p))
        if not rows: return []
        rows.sort(key=lambda x: x[3])
        pvals = [r[3] for r in rows]
        qvals = benjamini_hochberg(pvals)
        out = []
        for (term, k, m, p), q in zip(rows, qvals):
            out.append({"term": term, "overlap": k, "term_size": m, "p_value": p, "q_value": q})
        return out



    def build_full_graph(self):
        """用 rt 全量构建有向图（reg->target 全覆盖）"""
        self.G = nx.from_pandas_edgelist(self.rt, source="regulator",
                                        target="target", create_using=nx.DiGraph())


    def compute_metrics(self, approx: bool = True, k_sample: int = 500):
        """计算并缓存网络指标；大图时可用近似介数以提速"""
        if self.G is None:
            self.build_full_graph()
        G = self.G

        self.metrics = {
            "number_of_nodes": int(G.number_of_nodes()),
            "number_of_edges": int(G.number_of_edges()),
            "in_degree_distribution": [d for _, d in G.in_degree()],
            "out_degree_distribution": [d for _, d in G.out_degree()],
            "average_in_degree": float(np.mean([d for _, d in G.in_degree()])) if G.number_of_nodes() else 0.0,
            "average_out_degree": float(np.mean([d for _, d in G.out_degree()])) if G.number_of_nodes() else 0.0,
        }

        # 平均聚类系数要在无向图上算
        try:
            self.metrics["average_clustering_coefficient"] = float(
                nx.average_clustering(G.to_undirected()))
        except Exception:
            self.metrics["average_clustering_coefficient"] = None

        # 介数中心性：大图上建议用 k 源点近似
        try:
            if approx and G.number_of_nodes() > 3000:
                seeds = list(G.nodes())[:min(k_sample, G.number_of_nodes())]
                bc = nx.betweenness_centrality_subset(
                    G, sources=seeds, targets=list(G.nodes()))
            else:
                bc = nx.betweenness_centrality(G)
            self.metrics["betweenness_centrality"] = bc
        except Exception:
            self.metrics["betweenness_centrality"] = {}

        # 接近中心性
        try:
            self.metrics["closeness_centrality"] = nx.closeness_centrality(G)
        except Exception:
            self.metrics["closeness_centrality"] = {}

        # 特征向量中心性：可能不收敛，失败就置空
        try:
            self.metrics["eigenvector_centrality"] = nx.eigenvector_centrality(
                G, max_iter=500)
        except Exception:
            self.metrics["eigenvector_centrality"] = {}

        # 强连通分量
        scc = list(nx.strongly_connected_components(G))
        self.metrics["number_of_strongly_connected_components"] = len(scc)
        self.metrics["largest_strongly_connected_component"] = max(
            (len(c) for c in scc), default=0)

        # 可选拓扑排序示例（有环会失败）
        try:
            self.metrics["topological_sort_sample"] = list(
                nx.topological_sort(G))[:10]
        except Exception:
            self.metrics["topological_sort_sample"] = None

        # 关键节点集合（Top-N 摘要）
        def topn(d, n=10): return sorted(
            d.items(), key=lambda x: x[1], reverse=True)[:n]
        degree = dict(G.degree())
        self.key_nodes = {
            "high_degree_nodes": topn(degree, 10),
            "high_betweenness_nodes": topn(self.metrics.get("betweenness_centrality", {}), 10),
            "high_closeness_nodes": topn(self.metrics.get("closeness_centrality", {}), 10),
            "high_eigenvector_nodes": topn(self.metrics.get("eigenvector_centrality", {}), 10),
            "transcription_factors": sorted([g for g in self.nodes if g in self.reg_set]),
            "scc_top": sorted(scc, key=lambda x: len(x), reverse=True)[:5],
        }


    def common_two_step_paths(self, top_k: int = 20):
        """统计最常见的两跳通路 (u->v->w) 模式"""
        if self.G is None:
            self.build_full_graph()
        G = self.G
        counts = {}
        for u in G.nodes():
            for v in G.successors(u):
                for w in G.successors(v):
                    if u != w:
                        tpl = (u, v, w)
                        counts[tpl] = counts.get(tpl, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]


    def export_edges(self) -> pd.DataFrame:
        return self.rt.rename(columns={"regulator": "source", "target": "target"})


    def export_nodes_with_metrics(self) -> pd.DataFrame:
        """节点 + 注释 + 中心性指标"""
        if not self.metrics:
            self.compute_metrics()
        bc = self.metrics.get("betweenness_centrality", {})
        cc = self.metrics.get("closeness_centrality", {})
        ec = self.metrics.get("eigenvector_centrality", {})
        deg = dict(self.G.degree()) if self.G is not None else {}
        rows = []
        for gid in sorted(self.nodes):
            ann = self.get_annotation(gid)
            rows.append({
                "GeneID": gid,
                "Is_TF": ann.get("Is_TF"),
                "TF_Count": ann.get("TF_Count"),
                "Target_Count": ann.get("Target_Count"),
                "Degree": deg.get(gid, 0),
                "Betweenness": bc.get(gid, 0.0),
                "Closeness": cc.get(gid, 0.0),
                "Eigenvector": ec.get(gid, 0.0),
            })
        return pd.DataFrame(rows)
        # --------- Motifs (FFL / Bi-fan) ---------

    def motifs_ffl(self, rr_only: bool = False, limit: int = 50000):
        """
        Feed-Forward Loop: A->B->C with A->C.
        rr_only=True 时，仅保留 A、B、C 都是 TF 的 FFL；否则不限节点类型。
        返回: [{'A':A,'B':B,'C':C}, ...] 去重后按发现顺序输出（最多 limit 条）。
        """
        # 构建邻接与边集（用于 O(1) 查边）
        E = set(map(tuple, self.rt[["regulator", "target"]].itertuples(
            index=False, name=None)))
        out = []
        seen = set()
        for a, b in E:
            if rr_only and (a not in self.reg_set or b not in self.reg_set):
                continue
            # B 的下游是所有 C
            for c in self.reg_to_targets.get(b, set()):
                if rr_only and (c not in self.reg_set):
                    continue
                if (a, c) in E:
                    key = (a, b, c)
                    if key not in seen:
                        seen.add(key)
                        out.append({"A": a, "B": b, "C": c})
                        if len(out) >= limit:
                            return out
        return out

    def motifs_bifan(self, min_shared: int = 2, rr_only: bool = True, limit: int = 50000, expand_limit_per_pair: int = 200):
        """
        Bi-fan: 两个上游 (A,B) 共同调控至少两个下游 (C,D)。默认 rr_only=True 要求 A、B 为 TF。
        为控制规模：对每个 (A,B) 仅展开前 expand_limit_per_pair 个 (C,D) 组合；总体最多 limit 条。
        返回: [{'A':A,'B':B,'C':C,'D':D}, ...] （A<B，C<D 做规范化去重）。
        """
        from itertools import combinations
        pair2targets: Dict[tuple, Set[str]] = defaultdict(set)

        # 遍历每个 target，累积其共同调控的 regulator 对
        for t, regs in self.target_to_regs.items():
            regs_list = sorted([r for r in regs if (
                not rr_only or (r in self.reg_set))])
            if len(regs_list) < 2:
                continue
            for a, b in combinations(regs_list, 2):
                pair2targets[(a, b)].add(t)

        out = []
        seen = set()
        for (a, b), ts in pair2targets.items():
            if len(ts) < min_shared:
                continue
            # 可选：如果要求 C、D 也为 TF，可在此处过滤 ts = {t for t in ts if t in self.reg_set}
            cands = sorted(ts)
            emitted = 0
            # 展开 (C,D) 组合，但限制每对(A,B)最多 expand_limit_per_pair 个
            for i in range(len(cands)):
                for j in range(i + 1, len(cands)):
                    c, d = cands[i], cands[j]
                    quad = (a, b, c, d)
                    if quad not in seen:
                        seen.add(quad)
                        out.append({"A": a, "B": b, "C": c, "D": d})
                        emitted += 1
                        if len(out) >= limit or emitted >= expand_limit_per_pair:
                            break
                if len(out) >= limit or emitted >= expand_limit_per_pair:
                    break
            if len(out) >= limit:
                break
        return out
