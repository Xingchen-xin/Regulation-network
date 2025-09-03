
# Streptomyces 基因调控网络分析器 - 修复版本

import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pyvis.network import Network
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime
import json
from collections import Counter
import re

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style='whitegrid', font='WenQuanYi Zen Hei',
        rc={'axes.unicode_minus': False})

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("streptomyces_grn_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StreptomycesGRNAnalyzer:
    """Streptomyces coelicolor A32基因调控网络分析器"""

    def __init__(self, data_file, output_dir="grn_results"):
        """
        初始化分析器

        参数:
        data_file: 数据文件路径
        output_dir: 输出目录
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.raw_data = None
        self.processed_data = None
        self.network = None
        self.network_metrics = {}
        self.key_genes = {}
        self.enrichment_results = {}

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"初始化Streptomyces GRN分析器，数据文件: {data_file}")
        logger.info(f"结果将保存到: {output_dir}")

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        logger.info("开始数据加载和预处理...")

        try:
            # 读取CSV文件
            self.raw_data = pd.read_csv(self.data_file)
            logger.info(f"成功加载数据，共 {len(self.raw_data)} 条记录")

            # 检查必要的列
            required_columns = ['GeneID', 'Regulating_TFs', 'Regulates_Genes', 'Is_TF',
                               'Functional_Class', 'GO_Terms', 'Metabolic_Pathways']
            missing_columns = [
                col for col in required_columns if col not in self.raw_data.columns]

            if missing_columns:
                raise ValueError(f"数据文件缺少必要的列: {missing_columns}")

            # 处理缺失值
            self.raw_data.fillna('', inplace=True)

            # 统一基因ID格式
            self.raw_data['GeneID'] = self.raw_data['GeneID'].apply(
                self._standardize_gene_id)

            # 创建处理后的数据副本
            self.processed_data = self.raw_data.copy()

            logger.info("数据预处理完成")
            return True

        except Exception as e:
            logger.error(f"数据加载和预处理失败: {e}")
            return False

    def _standardize_gene_id(self, gene_id):
        """标准化基因ID格式"""
        if pd.isna(gene_id) or gene_id == '':
            return ''

        gene_id = str(gene_id).strip().upper()

        # 处理Streptomyces coelicolor基因ID格式 (如SCO2119 -> SCO_2119)
        if re.match(r'^SCO\d+$', gene_id):
            parts = gene_id[3:]
            if parts.isdigit():
                return f"SCO_{parts.zfill(4)}"

        return gene_id

    def build_regulatory_network(self):
        """构建基因调控网络"""
        logger.info("开始构建基因调控网络...")

        try:
            # 创建有向图
            self.network = nx.DiGraph()

            # 添加所有基因作为节点
            all_genes = set(self.processed_data['GeneID'])
            self.network.add_nodes_from(all_genes)

            # 添加节点属性
            for _, row in self.processed_data.iterrows():
                gene_id = row['GeneID']
                if gene_id in self.network.nodes:
                    self.network.nodes[gene_id]['is_tf'] = row['Is_TF']
                    self.network.nodes[gene_id]['functional_class'] = row['Functional_Class']
                    self.network.nodes[gene_id]['go_terms'] = row['GO_Terms']
                    self.network.nodes[gene_id]['metabolic_pathways'] = row['Metabolic_Pathways']

            # 添加调控关系边
            for _, row in self.processed_data.iterrows():
                target_gene = row['GeneID']

                # 处理调控该基因的转录因子
                regulating_tfs = str(row['Regulating_TFs']).split(',')
                for tf in regulating_tfs:
                    tf = tf.strip()
                    if tf and tf in self.network.nodes:
                        self.network.add_edge(
                            tf, target_gene, relationship='regulates')

                # 处理该基因调控的靶基因
                if row['Is_TF']:  # 只有转录因子才能调控其他基因
                    regulated_genes = str(row['Regulates_Genes']).split(',')
                    for target in regulated_genes:
                        target = target.strip()
                        if target and target in self.network.nodes:
                            self.network.add_edge(
                                target_gene, target, relationship='regulates')

            logger.info(
                f"成功构建调控网络，包含 {len(self.network.nodes)} 个节点和 {len(self.network.edges)} 条边")
            return True

        except Exception as e:
            logger.error(f"构建调控网络失败: {e}")
            return False

    def calculate_network_metrics(self):
        """计算网络拓扑指标"""
        logger.info("开始计算网络拓扑指标...")

        try:
            # 基本网络信息
            self.network_metrics['number_of_nodes'] = nx.number_of_nodes(
                self.network)
            self.network_metrics['number_of_edges'] = nx.number_of_edges(
                self.network)

            # 度分布指标
            in_degree_dict = dict(self.network.in_degree())
            out_degree_dict = dict(self.network.out_degree())

            self.network_metrics['in_degree_distribution'] = list(
                in_degree_dict.values())
            self.network_metrics['out_degree_distribution'] = list(
                out_degree_dict.values())

            # 计算平均度
            self.network_metrics['average_in_degree'] = np.mean(
                list(in_degree_dict.values())) if in_degree_dict else 0
            self.network_metrics['average_out_degree'] = np.mean(
                list(out_degree_dict.values())) if out_degree_dict else 0

            # 聚类系数（转换为无向网络）
            undirected_G = self.network.to_undirected()
            self.network_metrics['average_clustering_coefficient'] = nx.average_clustering(
                undirected_G)

            # 中心性指标
            self.network_metrics['betweenness_centrality'] = nx.betweenness_centrality(
                self.network)
            self.network_metrics['closeness_centrality'] = nx.closeness_centrality(
                self.network)

            try:
                self.network_metrics['eigenvector_centrality'] = nx.eigenvector_centrality(
                    self.network)
            except:
                self.network_metrics['eigenvector_centrality'] = None
                logger.warning("无法计算特征向量中心性（可能含有多个连通分量）")

            # 强连通分量
            scc = list(nx.strongly_connected_components(self.network))
            self.network_metrics['number_of_strongly_connected_components'] = len(
                scc)
            self.network_metrics['largest_strongly_connected_component'] = max(
                len(c) for c in scc) if scc else 0

            # 拓扑排序（仅对有向无环图有意义）
            try:
                self.network_metrics['topological_sort'] = list(
                    nx.topological_sort(self.network))[:10]
            except:
                self.network_metrics['topological_sort'] = None
                logger.warning("无法计算拓扑排序（网络可能含有环）")

            logger.info("网络拓扑指标计算完成")
            return True

        except Exception as e:
            logger.error(f"计算网络拓扑指标失败: {e}")
            return False

    def identify_key_genes_and_modules(self):
        """识别关键基因和调控模块"""
        logger.info("开始识别关键基因和调控模块...")

        try:
            # 获取度中心性高的节点
            degree_dict = dict(self.network.degree())
            self.key_genes['high_degree_nodes'] = sorted(
                degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]

            # 获取介数中心性高的节点
            betweenness_dict = self.network_metrics['betweenness_centrality']
            self.key_genes['high_betweenness_nodes'] = sorted(
                betweenness_dict.items(), key=lambda x: x[1], reverse=True)[:10]

            # 获取特征向量中心性高的节点（如果计算成功）
            if self.network_metrics['eigenvector_centrality']:
                eigenvector_dict = self.network_metrics['eigenvector_centrality']
                self.key_genes['high_eigenvector_nodes'] = sorted(
                    eigenvector_dict.items(), key=lambda x: x[1], reverse=True)[:10]

            # 识别转录因子
            tf_nodes = [node for node in self.network.nodes if self.network.nodes[node].get(
                'is_tf', False)]
            self.key_genes['transcription_factors'] = tf_nodes

            # 识别关键调控模块（强连通分量）
            scc = list(nx.strongly_connected_components(self.network))
            self.key_genes['strongly_connected_components'] = sorted(
                scc, key=lambda x: len(x), reverse=True)[:5]

            # 找出所有长度为2的路径（可能的三级调控关系）
            paths = []
            for node in self.network.nodes():
                for neighbor in self.network.successors(node):
                    for target in self.network.successors(neighbor):
                        if node != target and neighbor != target:
                            paths.append((node, neighbor, target))

            # 统计最常见的三级调控关系
            path_counts = {}
            for path in paths:
                path_tuple = tuple(path)
                path_counts[path_tuple] = path_counts.get(path_tuple, 0) + 1

            # 找出最常见的三级调控路径
            if path_counts:
                key_path = max(path_counts, key=path_counts.get)
                self.key_genes['key_regulatory_path'] = key_path
                self.key_genes['path_count'] = path_counts[key_path]
            else:
                self.key_genes['key_regulatory_path'] = None

            logger.info("关键基因和调控模块识别完成")
            return True

        except Exception as e:
            logger.error(f"识别关键基因和调控模块失败: {e}")
            return False

    def perform_functional_enrichment(self):
        """执行功能富集分析"""
        logger.info("开始执行功能富集分析...")

        try:
            # 获取关键基因列表
            key_gene_list = [gene for gene,
                _ in self.key_genes['high_degree_nodes']]

            # 获取所有基因
            all_genes = list(self.network.nodes)

            # 功能类别富集分析
            functional_classes = {}
            for gene in all_genes:
                func_class = self.network.nodes[gene].get(
                    'functional_class', 'Unknown')
                if func_class not in functional_classes:
                    functional_classes[func_class] = []
                functional_classes[func_class].append(gene)

            # 计算富集
            enrichment_results = {}
            for func_class, genes_in_class in functional_classes.items():
                # 计算在关键基因中的比例
                key_genes_in_class = set(key_gene_list) & set(genes_in_class)
                key_proportion = len(key_genes_in_class) / \
                                     len(key_gene_list) if key_gene_list else 0

                # 计算在所有基因中的比例
                all_proportion = len(genes_in_class) / \
                                     len(all_genes) if all_genes else 0

                # 计算富集分数
                enrichment_score = key_proportion / \
                    all_proportion if all_proportion > 0 else float('inf')

                enrichment_results[func_class] = {
                    'count_in_key': len(key_genes_in_class),
                    'proportion_in_key': key_proportion,
                    'count_in_background': len(genes_in_class),
                    'proportion_in_background': all_proportion,
                    'enrichment_score': enrichment_score
                }

            # 按富集分数排序
            self.enrichment_results['functional_class'] = dict(
                sorted(enrichment_results.items(),
                       key=lambda item: item[1]['enrichment_score'], reverse=True)
            )

            # GO Terms富集分析
            go_terms = {}
            for gene in all_genes:
                terms = str(self.network.nodes[gene].get(
                    'go_terms', '')).split(';')
                for term in terms:
                    term = term.strip()
                    if term:
                        if term not in go_terms:
                            go_terms[term] = []
                        go_terms[term].append(gene)

            # 计算GO富集
            go_enrichment = {}
            for go_term, genes_with_term in go_terms.items():
                # 计算在关键基因中的比例
                key_genes_with_term = set(key_gene_list) & set(genes_with_term)
                key_proportion = len(key_genes_with_term) / \
                                     len(key_gene_list) if key_gene_list else 0

                # 计算在所有基因中的比例
                all_proportion = len(genes_with_term) / \
                                     len(all_genes) if all_genes else 0

                # 计算富集分数
                enrichment_score = key_proportion / \
                    all_proportion if all_proportion > 0 else float('inf')

                go_enrichment[go_term] = {
                    'count_in_key': len(key_genes_with_term),
                    'proportion_in_key': key_proportion,
                    'count_in_background': len(genes_with_term),
                    'proportion_in_background': all_proportion,
                    'enrichment_score': enrichment_score
                }

            # 按富集分数排序，只保留前20个
            self.enrichment_results['go_terms'] = dict(
                sorted(go_enrichment.items(
                ), key=lambda item: item[1]['enrichment_score'], reverse=True)[:20]
            )

            # 代谢通路富集分析
            metabolic_pathways = {}
            for gene in all_genes:
                pathways = str(self.network.nodes[gene].get(
                    'metabolic_pathways', '')).split(';')
                for pathway in pathways:
                    pathway = pathway.strip()
                    if pathway:
                        if pathway not in metabolic_pathways:
                            metabolic_pathways[pathway] = []
                        metabolic_pathways[pathway].append(gene)

            # 计算代谢通路富集
            pathway_enrichment = {}
            for pathway, genes_in_pathway in metabolic_pathways.items():
                # 计算在关键基因中的比例
                key_genes_in_pathway = set(
                    key_gene_list) & set(genes_in_pathway)
                key_proportion = len(key_genes_in_pathway) / \
                                     len(key_gene_list) if key_gene_list else 0

                # 计算在所有基因中的比例
                all_proportion = len(genes_in_pathway) / \
                                     len(all_genes) if all_genes else 0

                # 计算富集分数
                enrichment_score = key_proportion / \
                    all_proportion if all_proportion > 0 else float('inf')

                pathway_enrichment[pathway] = {
                    'count_in_key': len(key_genes_in_pathway),
                    'proportion_in_key': key_proportion,
                    'count_in_background': len(genes_in_pathway),
                    'proportion_in_background': all_proportion,
                    'enrichment_score': enrichment_score
                }

            # 按富集分数排序
            self.enrichment_results['metabolic_pathways'] = dict(
                sorted(pathway_enrichment.items(),
                       key=lambda item: item[1]['enrichment_score'], reverse=True)
            )

            logger.info("功能富集分析完成")
            return True

        except Exception as e:
            logger.error(f"功能富集分析失败: {e}")
            return False

    def visualize_static_network(self, highlight_nodes=None, output_file=None):
        """创建静态网络可视化"""
        logger.info("开始创建静态网络可视化...")

        try:
            plt.figure(figsize=(15, 12))

            # 为不同的节点类型分配不同的颜色
            node_colors = []
            node_sizes = []

            # 获取度中心性
            degree_dict = dict(self.network.degree())

            # 计算平均度
            avg_degree = np.mean(list(degree_dict.values())
                                 ) if degree_dict else 0

            # 对节点着色和调整大小
            for node in self.network.nodes():
                degree = degree_dict[node]
                is_tf = self.network.nodes[node].get('is_tf', False)

                # 根据节点类型和度设置颜色
                if highlight_nodes and node in highlight_nodes:
                    node_colors.append('yellow')  # 高亮节点为黄色
                elif is_tf:
                    if degree > avg_degree * 2:
                        node_colors.append('red')  # 高度TF为红色
                    else:
                        node_colors.append('orange')  # 其他TF为橙色
                else:
                    if degree > avg_degree * 2:
                        node_colors.append('purple')  # 高度非TF为紫色
                    else:
                        node_colors.append('skyblue')  # 其他节点为蓝色

                # 根据节点度调整大小
                node_sizes.append(100 + degree * 20)

            # 使用spring布局
            pos = nx.spring_layout(self.network, k=0.8, iterations=50)

            # 绘制网络
            nx.draw_networkx_nodes(
                self.network, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
            nx.draw_networkx_edges(self.network, pos, alpha=0.3, arrowsize=10)

            # 添加标签（仅对高度节点添加）
            high_degree_nodes = [node for node, degree in degree_dict.items()
                               if degree > avg_degree * 1.5]
            labels = {node: node for node in high_degree_nodes}
            nx.draw_networkx_labels(
                self.network, pos, labels, font_size=8, font_weight='bold')

            # 设置标题和样式
            plt.title(f"Streptomyces coelicolor A32基因调控网络\n节点数: {self.network_metrics['number_of_nodes']}, 边数: {self.network_metrics['number_of_edges']}",
                     fontsize=16, pad=20)
            plt.tight_layout()

            # 如果指定了输出文件，则保存，否则显示
            if output_file is None:
                output_file = os.path.join(
                    self.output_dir, "static_network.png")

            plt.savefig(output_file, format='PNG', dpi=300)
            plt.close()

            logger.info(f"静态网络图已保存至: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"创建静态网络可视化失败: {e}")
            return None

    def create_interactive_network(self, highlight_nodes=None, output_file=None):
        """创建交互式网络可视化"""
        logger.info("开始创建交互式网络可视化...")

        try:
            # 创建PyVis网络对象
            nt = Network(height="1000px", width="100%",
                         directed=True, notebook=False)

            # 将NetworkX图转换为PyVis图
            nt.from_nx(self.network)

            # 为节点添加交互式信息
            for node in nt.nodes:
                # 获取节点属性
                node_id = node['id']
                node_degree = len(list(self.network.adj[node_id]))
                is_tf = self.network.nodes[node_id].get('is_tf', False)
                func_class = self.network.nodes[node_id].get(
                    'functional_class', 'Unknown')

                # 设置节点大小
                node['size'] = 10 + node_degree * 2

                # 设置节点颜色
                if highlight_nodes and node_id in highlight_nodes:
                    node['color'] = 'yellow'  # 高亮节点为黄色
                elif is_tf:
                    avg_out_degree = self.network_metrics.get(
                        'average_out_degree', 0)
                    if node_degree > avg_out_degree * 2:
                        node['color'] = 'red'  # 高度TF为红色
                    else:
                        node['color'] = 'orange'  # 其他TF为橙色
                else:
                    avg_in_degree = self.network_metrics.get(
                        'average_in_degree', 0)
                    if node_degree > avg_in_degree * 2:
                        node['color'] = 'purple'  # 高度非TF为紫色
                    else:
                        node['color'] = 'skyblue'  # 其他节点为蓝色

                # 添加工具提示
                node['title'] = f"基因: {node_id}<br>类型: {'转录因子' if is_tf else '非转录因子'}<br>功能类别: {func_class}<br>度: {node_degree}"

            # 为边添加颜色
            for edge in nt.edges:
                edge['color'] = '#888888'
                edge['width'] = 1

            # 更新布局选项
            options = """
            var options = {
                "physics": {
                    "barnesHut": {
                        "theta": 0.5,
                        "gravitationalConstant": -15000,
                        "centralGravity": 0.5
                    },
                    "maxVelocity": 50,
                    "minVelocity": 1
                },
                "interaction": {
                    "hover": true,
                    "tooltipDelay": 200,
                    "navigationButtons": true,
                    "keyboard": true
                }
            }
            """
            nt.set_options(options)

            # 如果指定了输出文件，则保存，否则使用默认文件名
            if output_file is None:
                output_file = os.path.join(
                    self.output_dir, "interactive_network.html")

            # 保存为HTML文件
            nt.show(output_file)

            logger.info(f"交互式网络已保存至: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"创建交互式网络可视化失败: {e}")
            return None

    def create_plotly_visualization(self, highlight_nodes=None, output_file=None):
        """使用Plotly创建高级网络可视化"""
        logger.info("开始创建Plotly高级网络可视化...")

        try:
            # 获取节点位置
            pos = nx.spring_layout(self.network, k=0.8, iterations=50)

            # 边的准备
            edge_x = []
            edge_y = []

            for edge in self.network.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.append(x0)
                edge_y.append(y0)
                edge_x.append(x1)
                edge_y.append(y1)
                edge_x.append(None)
                edge_y.append(None)

            # 创建边的轨迹
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            # 节点的准备
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []

            for node in self.network.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                # 获取节点属性
                degree = self.network.degree(node)
                is_tf = self.network.nodes[node].get('is_tf', False)
                func_class = self.network.nodes[node].get(
                    'functional_class', 'Unknown')

                # 节点颜色基于类型和度
                if highlight_nodes and node in highlight_nodes:
                    node_colors.append('yellow')  # 高亮节点为黄色
                elif is_tf:
                    avg_out_degree = self.network_metrics.get(
                        'average_out_degree', 0)
                    if degree > avg_out_degree * 2:
                        node_colors.append('red')  # 高度TF为红色
                    else:
                        node_colors.append('orange')  # 其他TF为橙色
                else:
                    avg_in_degree = self.network_metrics.get(
                        'average_in_degree', 0)
                    if degree > avg_in_degree * 2:
                        node_colors.append('purple')  # 高度非TF为紫色
                    else:
                        node_colors.append('skyblue')  # 其他节点为蓝色

                # 节点大小基于度
                node_sizes.append(10 + degree * 2)

                # 节点标签
                node_text.append(
                    f"{node}<br>类型: {'转录因子' if is_tf else '非转录因子'}<br>功能: {func_class}<br>度: {degree}")

            # 创建节点的散点图
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=False,
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=1)))

            # 创建图
            fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title=f"Streptomyces coelicolor A32基因调控网络<br>节点数: {len(self.network.nodes)}, 边数: {len(self.network.edges)}",
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper")],
                        xaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

            # 如果指定了输出文件，则保存，否则使用默认文件名
            if output_file is None:
                output_file = os.path.join(
                    self.output_dir, "plotly_network.html")

            # 保存为HTML文件
            fig.write_html(output_file)

            logger.info(f"Plotly高级网络可视化已保存至: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"创建Plotly高级网络可视化失败: {e}")
            return None

    def export_network_data(self):
        """导出网络数据"""
        logger.info("开始导出网络数据...")

        try:
            # 导出边数据
            edges_data = []
            for u, v, data in self.network.edges(data=True):
                edges_data.append({
                    'source': u,
                    'target': v,
                    'relationship': data.get('relationship', 'regulates')
                })

            edges_df = pd.DataFrame(edges_data)
            edges_file = os.path.join(self.output_dir, "network_edges.csv")
            edges_df.to_csv(edges_file, index=False)

            # 导出节点数据
            nodes_data = []
            for node in self.network.nodes():
                node_attrs = self.network.nodes[node]
                nodes_data.append({
                    'gene_id': node,
                    'is_tf': node_attrs.get('is_tf', False),
                    'functional_class': node_attrs.get('functional_class', ''),
                    'go_terms': node_attrs.get('go_terms', ''),
                    'metabolic_pathways': node_attrs.get('metabolic_pathways', ''),
                    'degree': self.network.degree(node),
                    'in_degree': self.network.in_degree(node),
                    'out_degree': self.network.out_degree(node),
                    'betweenness_centrality': self.network_metrics['betweenness_centrality'].get(node, 0),
                    'closeness_centrality': self.network_metrics['closeness_centrality'].get(node, 0),
                    'eigenvector_centrality': self.network_metrics['eigenvector_centrality'].get(node, 0) if self.network_metrics['eigenvector_centrality'] else 0
                })

            nodes_df = pd.DataFrame(nodes_data)
            nodes_file = os.path.join(self.output_dir, "network_nodes.csv")
            nodes_df.to_csv(nodes_file, index=False)

            # 导出GML格式
            gml_file = os.path.join(self.output_dir, "network.gml")
            nx.write_gml(self.network, gml_file)

            # 导出网络指标
            metrics_file = os.path.join(
                self.output_dir, "network_metrics.json")
            with open(metrics_file, 'w') as f:
                # 将numpy数组转换为列表以便JSON序列化
                metrics_to_save = {}
                for key, value in self.network_metrics.items():
                    if isinstance(value, np.ndarray):
                        metrics_to_save[key] = value.tolist()
                    elif isinstance(value, dict):
                        # 将集合转换为列表
                        clean_dict = {}
                        for k, v in value.items():
                            if isinstance(v, set):
                                clean_dict[k] = list(v)
                            else:
                                clean_dict[k] = v
                        metrics_to_save[key] = clean_dict
                    else:
                        metrics_to_save[key] = value

                json.dump(metrics_to_save, f, indent=4)

            # 导出关键基因
            key_genes_file = os.path.join(self.output_dir, "key_genes.json")
            with open(key_genes_file, 'w') as f:
                # 将集合转换为列表
                clean_key_genes = {}
                for k, v in self.key_genes.items():
                    if isinstance(v, set):
                        clean_key_genes[k] = list(v)
                    else:
                        clean_key_genes[k] = v
                json.dump(clean_key_genes, f, indent=4)

            # 导出富集分析结果
            enrichment_file = os.path.join(
                self.output_dir, "enrichment_results.json")
            with open(enrichment_file, 'w') as f:
                json.dump(self.enrichment_results, f, indent=4)

            logger.info(f"网络数据已导出至: {self.output_dir}")
            return True

        except Exception as e:
            logger.error(f"导出网络数据失败: {e}")
            return False

    def create_summary_report(self):
        """创建分析摘要报告"""
        logger.info("开始创建分析摘要报告...")

        try:
            report_file = os.path.join(self.output_dir, "analysis_report.md")

            with open(report_file, 'w') as f:
                f.write("# Streptomyces coelicolor A32基因调控网络分析报告\n\n")
                f.write(
                    f"**分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # 网络基本信息
                f.write("## 1. 网络基本信息\n\n")
                f.write(f"- 节点数: {self.network_metrics['number_of_nodes']}\n")
                f.write(f"- 边数: {self.network_metrics['number_of_edges']}\n")
                f.write(
                    f"- 平均入度: {self.network_metrics['average_in_degree']:.2f}\n")
                f.write(
                    f"- 平均出度: {self.network_metrics['average_out_degree']:.2f}\n")
                f.write(
                    f"- 平均聚类系数: {self.network_metrics['average_clustering_coefficient']:.2f}\n")
                f.write(
                    f"- 强连通分量数: {self.network_metrics['number_of_strongly_connected_components']}\n")
                f.write(
                    f"- 最大强连通分量大小: {self.network_metrics['largest_strongly_connected_component']}\n\n")

                # 关键基因
                f.write("## 2. 关键基因\n\n")
                f.write("### 2.1 高度基因\n\n")
                f.write("| 基因ID | 度 |\n")
                f.write("|--------|----|\n")
                for gene, degree in self.key_genes['high_degree_nodes'][:10]:
                    f.write(f"| {gene} | {degree} |\n")

                f.write("\n### 2.2 高介数中心性基因\n\n")
                f.write("| 基因ID | 介数中心性 |\n")
                f.write("|--------|------------|\n")
                for gene, centrality in self.key_genes['high_betweenness_nodes'][:10]:
                    f.write(f"| {gene} | {centrality:.4f} |\n")

                if self.network_metrics['eigenvector_centrality']:
                    f.write("\n### 2.3 高特征向量中心性基因\n\n")
                    f.write("| 基因ID | 特征向量中心性 |\n")
                    f.write("|--------|--------------|\n")
                    for gene, centrality in self.key_genes['high_eigenvector_nodes'][:10]:
                        f.write(f"| {gene} | {centrality:.4f} |\n")

                f.write(f"\n### 2.4 转录因子\n\n")
                f.write(
                    f"网络中共有 {len(self.key_genes['transcription_factors'])} 个转录因子。\n")

                f.write("\n### 2.5 关键调控路径\n\n")
                if self.key_genes['key_regulatory_path']:
                    path = " → ".join(self.key_genes['key_regulatory_path'])
                    f.write(
                        f"最频繁的三级调控路径: {path} (出现次数: {self.key_genes['path_count']})\n")
                else:
                    f.write("未发现明显的三级调控路径。\n")
                # 功能富集分析
                f.write("\n## 3. 功能富集分析\n\n")

                f.write("### 3.1 功能类别富集\n\n")
                f.write("| 功能类别 | 关键基因数 | 背景基因数 | 富集分数 |\n")
                f.write("|----------|------------|------------|----------|\n")
                for func_class, data in list(self.enrichment_results['functional_class'].items())[:10]:
                    f.write(
                        f"| {func_class} | {data['count_in_key']} | {data['count_in_background']} | {data['enrichment_score']:.2f} |\n")

                f.write("\n### 3.2 GO Terms富集\n\n")
                f.write("| GO Term | 关键基因数 | 背景基因数 | 富集分数 |\n")
                f.write("|---------|------------|------------|----------|\n")
                for go_term, data in list(self.enrichment_results['go_terms'].items())[:10]:
                    f.write(
                        f"| {go_term} | {data['count_in_key']} | {data['count_in_background']} | {data['enrichment_score']:.2f} |\n")

                f.write("\n### 3.3 代谢通路富集\n\n")
                f.write("| 代谢通路 | 关键基因数 | 背景基因数 | 富集分数 |\n")
                f.write("|----------|------------|------------|----------|\n")
                for pathway, data in list(self.enrichment_results['metabolic_pathways'].items())[:10]:
                    f.write(
                        f"| {pathway} | {data['count_in_key']} | {data['count_in_background']} | {data['enrichment_score']:.2f} |\n")

                # 结论
                f.write("\n## 4. 结论\n\n")
                f.write(
                    "本分析对Streptomyces coelicolor A32的基因调控网络进行了全面分析，识别了关键基因和调控模块，并进行了功能富集分析。")
                f.write("这些结果为进一步理解Streptomyces coelicolor A32的基因调控机制提供了重要参考。\n")

            logger.info(f"分析摘要报告已保存至: {report_file}")
            return report_file

        except Exception as e:
            logger.error(f"创建分析摘要报告失败: {e}")
            return None

    def run_full_analysis(self):
        """运行完整的分析流程"""
        logger.info("开始运行完整的基因调控网络分析流程...")

        try:
            # 1. 数据加载和预处理
            if not self.load_and_preprocess_data():
                return False

            # 2. 构建调控网络
            if not self.build_regulatory_network():
                return False

            # 3. 计算网络指标
            if not self.calculate_network_metrics():
                return False

            # 4. 识别关键基因和模块
            if not self.identify_key_genes_and_modules():
                return False

            # 5. 功能富集分析
            if not self.perform_functional_enrichment():
                return False

            # 6. 创建可视化
            self.visualize_static_network()
            self.create_interactive_network()
            self.create_plotly_visualization()

            # 7. 导出数据
            self.export_network_data()

            # 8. 创建摘要报告
            self.create_summary_report()

            logger.info("完整的基因调控网络分析流程已完成！")
            return True

        except Exception as e:
            logger.error(f"分析流程失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    # 设置命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python streptomyces_grn_analyzer.py <数据文件路径> [输出目录]")
        sys.exit(1)

    data_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "grn_results"

    # 检查数据文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 数据文件 {data_file} 不存在")
        sys.exit(1)

    # 创建分析器并运行分析
    analyzer = StreptomycesGRNAnalyzer(data_file, output_dir)
    success = analyzer.run_full_analysis()

    if success:
        print(f"分析完成！结果已保存到 {output_dir} 目录")
    else:
        print("分析失败，请检查日志文件获取详细信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
