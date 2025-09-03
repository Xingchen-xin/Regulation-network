#Added by GLM

import os
import re
import json
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set, Optional, Union
import base64
import io
from sklearn.preprocessing import LabelEncoder
import community as community_louvain
from flask import Flask, render_template, request, jsonify, send_file
import webbrowser
import threading
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style='whitegrid', font='WenQuanYi Zen Hei', rc={'axes.unicode_minus': False})

class RegulatoryNetworkAnalyzer:
    """调控网络分析器类，用于处理Streptomyces coelicolor A32的调控网络数据"""
    
    def __init__(self, data_file: str = None):
        """
        初始化网络分析器
        
        参数:
            data_file: 数据文件路径，可以是TSV或CSV格式
        """
        self.data_file = data_file
        self.raw_data = None
        self.processed_data = None
        self.network = nx.DiGraph()
        self.node_attributes = {}
        self.scc_mapping = {}
        self.go_terms = {}
        self.metabolic_pathways = {}
        self.gene_functions = {}
        self.language = 'zh'  # 默认中文
        
        # 如果提供了数据文件，则加载数据
        if data_file:
            self.load_data(data_file)
    
    def load_data(self, file_path: str) -> None:
        """
        加载并解析数据文件
        
        参数:
            file_path: 数据文件路径
        """
        try:
            # 自动检测分隔符和编码
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # 尝试不同编码
            encodings = ['utf-8', 'gbk', 'latin1', 'iso-8859-1']
            decoded_content = None
            
            for encoding in encodings:
                try:
                    decoded_content = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if decoded_content is None:
                raise ValueError("无法解码文件，请检查文件编码")
            
            # 检测分隔符
            first_line = decoded_content.split('\n')[0]
            if '\t' in first_line:
                separator = '\t'
            elif ',' in first_line:
                separator = ','
            else:
                separator = '\t'  # 默认使用制表符
            
            # 读取数据
            self.raw_data = pd.read_csv(file_path, sep=separator, encoding=encoding)
            
            # 处理数据
            self.process_data()
            
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            raise
    
    def process_data(self) -> None:
        """处理原始数据，包括清洗、转换和标准化"""
        if self.raw_data is None:
            raise ValueError("没有可用的原始数据，请先加载数据")
        
        # 创建处理后的数据副本
        self.processed_data = self.raw_data.copy()
        
        # 1. 处理列名
        # 标准化列名（小写，去除空格）
        self.processed_data.columns = [col.strip().lower() for col in self.processed_data.columns]
        
        # 确保必要的列存在
        required_columns = ['regulator', 'target']
        for col in required_columns:
            if col not in self.processed_data.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 2. 处理缺失值
        # 删除regulator或target为空的行
        self.processed_data.dropna(subset=['regulator', 'target'], inplace=True)
        
        # 3. 统一基因位号大小写（sco→SCO）
        self.processed_data['regulator'] = self.processed_data['regulator'].apply(
            lambda x: re.sub(r'\bsco(\d+)\b', r'SCO\1', str(x))
        )
        self.processed_data['target'] = self.processed_data['target'].apply(
            lambda x: re.sub(r'\bsco(\d+)\b', r'SCO\1', str(x))
        )
        
        # 4. 拆分target基因（按", ; 空格"拆分）
        # 创建一个新的DataFrame来存储拆分后的数据
        expanded_data = []
        
        for _, row in self.processed_data.iterrows():
            targets = re.split(r'[,; ]+', str(row['target']))
            for target in targets:
                if target.strip():  # 忽略空字符串
                    new_row = row.copy()
                    new_row['target'] = target.strip()
                    expanded_data.append(new_row)
        
        self.processed_data = pd.DataFrame(expanded_data)
        
        # 5. 处理重复数据
        # 保留regulator-target对的唯一组合
        self.processed_data.drop_duplicates(subset=['regulator', 'target'], keep='first', inplace=True)
        
        # 6. 重置索引
        self.processed_data.reset_index(drop=True, inplace=True)
        
        # 7. 构建网络
        self.build_network()
    
    def build_network(self) -> None:
        """基于处理后的数据构建调控网络"""
        if self.processed_data is None:
            raise ValueError("没有可用的处理数据，请先处理数据")
        
        # 创建有向图
        self.network = nx.DiGraph()
        
        # 添加节点和边
        for _, row in self.processed_data.iterrows():
            regulator = str(row['regulator'])
            target = str(row['target'])
            
            # 添加节点（如果不存在）
            if regulator not in self.network:
                self.network.add_node(regulator, type='regulator')
            if target not in self.network:
                self.network.add_node(target, type='target')
            
            # 添加边
            self.network.add_edge(regulator, target)
            
            # 如果target也是regulator，则更新其类型
            if target in self.processed_data['regulator'].values:
                self.network.nodes[target]['type'] = 'regulator'
        
        # 标记Reg→Reg边
        for u, v in self.network.edges():
            if self.network.nodes[u]['type'] == 'regulator' and self.network.nodes[v]['type'] == 'regulator':
                self.network[u][v]['reg_to_reg'] = True
            else:
                self.network[u][v]['reg_to_reg'] = False
        
        # 计算节点属性
        self.calculate_node_attributes()
        
        # 计算强连通分量
        self.calculate_scc()
    
    def calculate_node_attributes(self) -> None:
        """计算节点的属性，如入度、出度等"""
        for node in self.network.nodes():
            self.node_attributes[node] = {
                'in_degree': self.network.in_degree(node),
                'out_degree': self.network.out_degree(node),
                'total_degree': self.network.degree(node),
                'type': self.network.nodes[node]['type']
            }
    
    def calculate_scc(self) -> None:
        """计算强连通分量(SCC)"""
        scc = list(nx.strongly_connected_components(self.network))
        
        # 为每个节点分配SCC ID
        for i, component in enumerate(scc):
            for node in component:
                self.scc_mapping[node] = i
    
    def analyze_network_topology(self) -> Dict:
        """分析网络拓扑特性"""
        if not self.network:
            raise ValueError("网络未构建，请先构建网络")
        
        # 基本网络属性
        num_nodes = self.network.number_of_nodes()
        num_edges = self.network.number_of_edges()
        
        # 密度
        density = nx.density(self.network)
        
        # 连通性
        is_weakly_connected = nx.is_weakly_connected(self.network)
        is_strongly_connected = nx.is_strongly_connected(self.network)
        
        # 强连通分量
        num_scc = len(self.scc_mapping)
        largest_scc_size = max(len(set([n for n in self.network.nodes() if self.scc_mapping.get(n) == i])) 
                             for i in set(self.scc_mapping.values()))
        
        # 平均路径长度（仅对弱连通图计算）
        if is_weakly_connected:
            avg_path_length = nx.average_shortest_path_length(self.network.to_undirected())
        else:
            avg_path_length = None
        
        # 聚类系数
        avg_clustering = nx.average_clustering(self.network.to_undirected())
        
        # 度分布
        in_degrees = [d for n, d in self.network.in_degree()]
        out_degrees = [d for n, d in self.network.out_degree()]
        
        # 中心性指标
        # 计算部分中心性指标（对于大型网络，计算所有中心性指标可能很耗时）
        try:
            # 入度中心性
            in_degree_centrality = nx.in_degree_centrality(self.network)
            # 出度中心性
            out_degree_centrality = nx.out_degree_centrality(self.network)
            # 紧密中心性（仅对弱连通图计算）
            if is_weakly_connected:
                closeness_centrality = nx.closeness_centrality(self.network)
            else:
                closeness_centrality = None
            # PageRank
            pagerank = nx.pagerank(self.network)
        except Exception as e:
            print(f"计算中心性指标时出错: {str(e)}")
            in_degree_centrality = None
            out_degree_centrality = None
            closeness_centrality = None
            pagerank = None
        
        # 社区检测（使用Louvain算法）
        try:
            # 转换为无向图进行社区检测
            undirected_network = self.network.to_undirected()
            communities = community_louvain.best_partition(undirected_network)
            num_communities = len(set(communities.values()))
            modularity = community_louvain.modularity(communities, undirected_network)
        except Exception as e:
            print(f"社区检测时出错: {str(e)}")
            communities = None
            num_communities = None
            modularity = None
        
        # 返回分析结果
        return {
            'basic_properties': {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'density': density,
                'is_weakly_connected': is_weakly_connected,
                'is_strongly_connected': is_strongly_connected
            },
            'connectivity': {
                'num_scc': num_scc,
                'largest_scc_size': largest_scc_size,
                'avg_path_length': avg_path_length
            },
            'clustering': {
                'avg_clustering': avg_clustering
            },
            'degree_distribution': {
                'in_degrees': in_degrees,
                'out_degrees': out_degrees
            },
            'centrality': {
                'in_degree_centrality': in_degree_centrality,
                'out_degree_centrality': out_degree_centrality,
                'closeness_centrality': closeness_centrality,
                'pagerank': pagerank
            },
            'community': {
                'communities': communities,
                'num_communities': num_communities,
                'modularity': modularity
            }
        }
    
    def load_annotation_data(self, go_file: str = None, pathway_file: str = None, function_file: str = None) -> None:
        """
        加载注释数据，包括GO Terms、代谢通路和基因功能
        
        参数:
            go_file: GO Terms注释文件路径
            pathway_file: 代谢通路注释文件路径
            function_file: 基因功能注释文件路径
        """
        # 加载GO Terms
        if go_file:
            try:
                with open(go_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            gene_id = parts[0]
                            go_terms = parts[1:]
                            self.go_terms[gene_id] = go_terms
            except Exception as e:
                print(f"加载GO Terms时出错: {str(e)}")
        
        # 加载代谢通路
        if pathway_file:
            try:
                with open(pathway_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            gene_id = parts[0]
                            pathways = parts[1:]
                            self.metabolic_pathways[gene_id] = pathways
            except Exception as e:
                print(f"加载代谢通路时出错: {str(e)}")
        
        # 加载基因功能
        if function_file:
            try:
                with open(function_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            gene_id = parts[0]
                            function = parts[1]
                            self.gene_functions[gene_id] = function
            except Exception as e:
                print(f"加载基因功能时出错: {str(e)}")
    
    def get_node_info(self, node_id: str) -> Dict:
        """
        获取节点的详细信息
        
        参数:
            node_id: 节点ID
            
        返回:
            包含节点详细信息的字典
        """
        if node_id not in self.network:
            return {}
        
        # 基本信息
        info = {
            'id': node_id,
            'type': self.network.nodes[node_id]['type'],
            'in_degree': self.network.in_degree(node_id),
            'out_degree': self.network.out_degree(node_id),
            'total_degree': self.network.degree(node_id),
            'scc_id': self.scc_mapping.get(node_id, -1)
        }
        
        # 上游调控因子
        predecessors = list(self.network.predecessors(node_id))
        info['regulators'] = predecessors
        
        # 下游靶基因
        successors = list(self.network.successors(node_id))
        info['targets'] = successors
        
        # 注释信息
        if node_id in self.go_terms:
            info['go_terms'] = self.go_terms[node_id]
        
        if node_id in self.metabolic_pathways:
            info['pathways'] = self.metabolic_pathways[node_id]
        
        if node_id in self.gene_functions:
            info['function'] = self.gene_functions[node_id]
        
        return info
    
    def get_subnetwork(self, nodes: List[str], depth: int = 1) -> nx.DiGraph:
        """
        获取子网络，包括指定节点及其邻域
        
        参数:
            nodes: 起始节点列表
            depth: 邻域深度
            
        返回:
            子网络图
        """
        if not nodes or depth < 1:
            return nx.DiGraph()
        
        # 确保所有节点都在网络中
        valid_nodes = [node for node in nodes if node in self.network]
        if not valid_nodes:
            return nx.DiGraph()
        
        # 创建子网络
        subgraph_nodes = set(valid_nodes)
        
        # 根据深度扩展节点
        for _ in range(depth):
            new_nodes = set()
            for node in subgraph_nodes:
                # 添加上游节点
                new_nodes.update(self.network.predecessors(node))
                # 添加下游节点
                new_nodes.update(self.network.successors(node))
            
            subgraph_nodes.update(new_nodes)
        
        # 创建子图
        subgraph = self.network.subgraph(subgraph_nodes).copy()
        
        return subgraph
    
    def find_shortest_path(self, source: str, target: str) -> List[str]:
        """
        查找两个节点之间的最短路径
        
        参数:
            source: 源节点
            target: 目标节点
            
        返回:
            最短路径节点列表，如果不存在路径则返回空列表
        """
        if source not in self.network or target not in self.network:
            return []
        
        try:
            path = nx.shortest_path(self.network, source=source, target=target)
            return path
        except nx.NetworkXNoPath:
            return []
    
    def get_upstream_downstream_table(self, node_id: str) -> Dict:
        """
        获取节点的上下游基因表格数据
        
        参数:
            node_id: 节点ID
            
        返回:
            包含上下游基因表格数据的字典
        """
        if node_id not in self.network:
            return {'upstream': [], 'downstream': []}
        
        # 上游调控因子
        upstream_data = []
        for regulator in self.network.predecessors(node_id):
            regulator_info = self.get_node_info(regulator)
            upstream_data.append({
                'id': regulator,
                'type': regulator_info.get('type', ''),
                'function': regulator_info.get('function', ''),
                'go_terms': ', '.join(regulator_info.get('go_terms', [])),
                'pathways': ', '.join(regulator_info.get('pathways', []))
            })
        
        # 下游靶基因
        downstream_data = []
        for target in self.network.successors(node_id):
            target_info = self.get_node_info(target)
            downstream_data.append({
                'id': target,
                'type': target_info.get('type', ''),
                'function': target_info.get('function', ''),
                'go_terms': ', '.join(target_info.get('go_terms', [])),
                'pathways': ', '.join(target_info.get('pathways', []))
            })
        
        return {
            'upstream': upstream_data,
            'downstream': downstream_data
        }
    
    def export_network_to_png(self, filename: str, layout: str = 'spring', figsize: Tuple[int, int] = (20, 20)) -> str:
        """
        将网络图导出为PNG文件
        
        参数:
            filename: 输出文件名
            layout: 布局算法，可选'spring', 'circular', 'random', 'shell'
            figsize: 图像大小
            
        返回:
            输出文件的绝对路径
        """
        if not self.network:
            raise ValueError("网络未构建，请先构建网络")
        
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(self.network, k=0.15, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.network)
        elif layout == 'random':
            pos = nx.random_layout(self.network)
        elif layout == 'shell':
            pos = nx.shell_layout(self.network)
        else:
            pos = nx.spring_layout(self.network)
        
        # 绘制节点
        node_colors = []
        for node in self.network.nodes():
            if self.network.nodes[node]['type'] == 'regulator':
                node_colors.append('red')
            else:
                node_colors.append('blue')
        
        nx.draw_networkx_nodes(self.network, pos, node_size=100, node_color=node_colors, alpha=0.8)
        
        # 绘制边
        edge_colors = []
        for u, v in self.network.edges():
            if self.network[u][v].get('reg_to_reg', False):
                edge_colors.append('red')
            else:
                edge_colors.append('gray')
        
        nx.draw_networkx_edges(self.network, pos, width=1, alpha=0.5, edge_color=edge_colors)
        
        # 绘制标签（仅对小型网络）
        if self.network.number_of_nodes() <= 100:
            nx.draw_networkx_labels(self.network, pos, font_size=8)
        
        # 添加图例
        regulator_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Regulator')
        target_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Target')
        reg_reg_patch = plt.Line2D([0], [0], color='red', linewidth=2, label='Reg→Reg')
        reg_target_patch = plt.Line2D([0], [0], color='gray', linewidth=2, label='Reg→Target')
        plt.legend(handles=[regulator_patch, target_patch, reg_reg_patch, reg_target_patch])
        
        # 添加标题
        plt.title("Streptomyces coelicolor A32 Regulatory Network")
        
        # 保存图像
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return os.path.abspath(filename)
    
    def export_table_to_csv(self, data: List[Dict], filename: str) -> str:
        """
        将表格数据导出为CSV文件
        
        参数:
            data: 表格数据
            filename: 输出文件名
            
        返回:
            输出文件的绝对路径
        """
        if not data:
            raise ValueError("没有可导出的数据")
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 保存为CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        
        return os.path.abspath(filename)
    
    def prepare_cytoscape_data(self) -> Dict:
        """
        准备Cytoscape.js格式的网络数据
        
        返回:
            包含节点和边数据的字典
        """
        nodes = []
        edges = []
        
        # 添加节点
        for node in self.network.nodes():
            node_data = {
                'data': {
                    'id': node,
                    'type': self.network.nodes[node]['type'],
                    'in_degree': self.network.in_degree(node),
                    'out_degree': self.network.out_degree(node),
                    'total_degree': self.network.degree(node),
                    'scc_id': self.scc_mapping.get(node, -1)
                }
            }
            
            # 添加注释信息
            if node in self.gene_functions:
                node_data['data']['function'] = self.gene_functions[node]
            
            if node in self.go_terms:
                node_data['data']['go_terms'] = ', '.join(self.go_terms[node])
            
            if node in self.metabolic_pathways:
                node_data['data']['pathways'] = ', '.join(self.metabolic_pathways[node])
            
            nodes.append(node_data)
        
        # 添加边
        for u, v, data in self.network.edges(data=True):
            edge_data = {
                'data': {
                    'id': f"{u}-{v}",
                    'source': u,
                    'target': v,
                    'reg_to_reg': data.get('reg_to_reg', False)
                }
            }
            edges.append(edge_data)
        
        return {
            'nodes': nodes,
            'edges': edges
        }


class RegulatoryNetworkApp:
    """调控网络Web应用类"""
    
    def __init__(self, data_file: str = None, go_file: str = None, pathway_file: str = None, function_file: str = None):
        """
        初始化Web应用
        
        参数:
            data_file: 调控网络数据文件路径
            go_file: GO Terms注释文件路径
            pathway_file: 代谢通路注释文件路径
            function_file: 基因功能注释文件路径
        """
        self.app = Flask(__name__)
        self.analyzer = RegulatoryNetworkAnalyzer(data_file)
        
        # 加载注释数据
        if go_file or pathway_file or function_file:
            self.analyzer.load_annotation_data(go_file, pathway_file, function_file)
        
        # 设置路由
        self.setup_routes()
    
    def setup_routes(self):
        """设置Flask应用的路由"""
        
        @self.app.route('/')
        def index():
            """主页"""
            return render_template('index.html')
        
        @self.app.route('/api/network')
        def get_network():
            """获取网络数据"""
            try:
                network_data = self.analyzer.prepare_cytoscape_data()
                return jsonify(network_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/node/<node_id>')
        def get_node_info(node_id):
            """获取节点信息"""
            try:
                node_info = self.analyzer.get_node_info(node_id)
                return jsonify(node_info)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/subnetwork', methods=['POST'])
        def get_subnetwork():
            """获取子网络"""
            try:
                data = request.json
                nodes = data.get('nodes', [])
                depth = data.get('depth', 1)
                
                subgraph = self.analyzer.get_subnetwork(nodes, depth)
                
                # 转换为Cytoscape格式
                nodes = []
                edges = []
                
                for node in subgraph.nodes():
                    node_data = {
                        'data': {
                            'id': node,
                            'type': subgraph.nodes[node]['type'],
                            'in_degree': subgraph.in_degree(node),
                            'out_degree': subgraph.out_degree(node),
                            'total_degree': subgraph.degree(node),
                            'scc_id': self.analyzer.scc_mapping.get(node, -1)
                        }
                    }
                    
                    # 添加注释信息
                    if node in self.analyzer.gene_functions:
                        node_data['data']['function'] = self.analyzer.gene_functions[node]
                    
                    if node in self.analyzer.go_terms:
                        node_data['data']['go_terms'] = ', '.join(self.analyzer.go_terms[node])
                    
                    if node in self.analyzer.metabolic_pathways:
                        node_data['data']['pathways'] = ', '.join(self.analyzer.metabolic_pathways[node])
                    
                    nodes.append(node_data)
                
                for u, v, data in subgraph.edges(data=True):
                    edge_data = {
                        'data': {
                            'id': f"{u}-{v}",
                            'source': u,
                            'target': v,
                            'reg_to_reg': data.get('reg_to_reg', False)
                        }
                    }
                    edges.append(edge_data)
                
                return jsonify({
                    'nodes': nodes,
                    'edges': edges
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/shortest_path', methods=['POST'])
        def get_shortest_path():
            """获取最短路径"""
            try:
                data = request.json
                source = data.get('source')
                target = data.get('target')
                
                path = self.analyzer.find_shortest_path(source, target)
                
                return jsonify({
                    'path': path
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/table/<node_id>')
        def get_table(node_id):
            """获取上下游基因表格"""
            try:
                table_data = self.analyzer.get_upstream_downstream_table(node_id)
                return jsonify(table_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/export/png', methods=['POST'])
        def export_png():
            """导出网络图为PNG"""
            try:
                data = request.json
                layout = data.get('layout', 'spring')
                
                # 生成临时文件名
                timestamp = int(time.time())
                filename = f"network_{timestamp}.png"
                
                # 导出图像
                filepath = self.analyzer.export_network_to_png(filename, layout)
                
                # 读取图像并转换为base64
                with open(filepath, 'rb') as f:
                    image_data = f.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                # 删除临时文件
                os.remove(filepath)
                
                return jsonify({
                    'image': image_base64,
                    'filename': filename
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/export/csv', methods=['POST'])
        def export_csv():
            """导出表格为CSV"""
            try:
                data = request.json
                table_data = data.get('data', [])
                table_type = data.get('type', 'upstream')
                node_id = data.get('node_id', 'unknown')
                
                # 生成临时文件名
                timestamp = int(time.time())
                filename = f"{node_id}_{table_type}_{timestamp}.csv"
                
                # 导出CSV
                filepath = self.analyzer.export_table_to_csv(table_data, filename)
                
                # 读取文件并转换为base64
                with open(filepath, 'rb') as f:
                    file_data = f.read()
                    file_base64 = base64.b64encode(file_data).decode('utf-8')
                
                # 删除临时文件
                os.remove(filepath)
                
                return jsonify({
                    'file': file_base64,
                    'filename': filename
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/topology')
        def get_topology():
            """获取网络拓扑分析结果"""
            try:
                topology = self.analyzer.analyze_network_topology()
                return jsonify(topology)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/language/<lang>')
        def set_language(lang):
            """设置界面语言"""
            if lang in ['zh', 'en']:
                self.analyzer.language = lang
                return jsonify({'success': True})
            else:
                return jsonify({'error': 'Unsupported language'}), 400
    
    def run(self, host='127.0.0.1', port=5000, debug=False, open_browser=True):
        """运行Web应用"""
        if open_browser:
            # 在新线程中打开浏览器
            def open_browser_delayed():
                time.sleep(1.5)  # 等待服务器启动
                webbrowser.open(f'http://{host}:{port}')
            
            threading.Thread(target=open_browser_delayed).start()
        
        self.app.run(host=host, port=port, debug=debug)


def create_html_template():
    """创建HTML模板文件"""
    template_dir = 'templates'
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    index_html = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Streptomyces coelicolor A32 调控网络分析</title>
    
    <!-- 引入Cytoscape.js -->
    <script src="https://autoglm-api.zhipuai.cn/ppt-proxy/https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.19.0/cytoscape.min.js"></script>
    
    <!-- 引入Cytoscape.js布局扩展 -->
    <script src="https://autoglm-api.zhipuai.cn/ppt-proxy/https://cdnjs.cloudflare.com/ajax/libs/dagre/0.5.1/dagre.min.js"></script>
    <script src="https://autoglm-api.zhipuai.cn/ppt-proxy/https://cdnjs.cloudflare.com/ajax/libs/cytoscape-dagre/2.3.2/cytoscape-dagre.min.js"></script>
    <script src="https://autoglm-api.zhipuai.cn/ppt-proxy/https://cdnjs.cloudflare.com/ajax/libs/cose-base/1.0.0/cose-base.min.js"></script>
    <script src="https://autoglm-api.zhipuai.cn/ppt-proxy/https://cdnjs.cloudflare.com/ajax/libs/cytoscape-cose-bilkent/4.1.0/cytoscape-cose-bilkent.min.js"></script>
    <script src="https://autoglm-api.zhipuai.cn/ppt-proxy/https://cdnjs.cloudflare.com/ajax/libs/cytoscape-cola/2.5.1/cytoscape-cola.min.js"></script>
    
    <!-- 引入Bootstrap -->
    <link href="https://autoglm-api.zhipuai.cn/ppt-proxy/https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://autoglm-api.zhipuai.cn/ppt-proxy/https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- 引入Font Awesome -->
    <link rel="stylesheet" href="https://autoglm-api.zhipuai.cn/ppt-proxy/https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    
    <!-- 引入DataTables -->
    <link rel="stylesheet" type="text/css" href="https://autoglm-api.zhipuai.cn/ppt-proxy/https://cdn.datatables.net/1.11.5/css/dataTables.bootstrap5.min.css">
    <script type="text/javascript" src="https://autoglm-api.zhipuai.cn/ppt-proxy/https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
    <script type="text/javascript" src="https://autoglm-api.zhipuai.cn/ppt-proxy/https://cdn.datatables.net/1.11.5/js/dataTables.bootstrap5.min.js"></script>
    
    <!-- 自定义样式 -->
    <style>
        body {
            font-family: 'WenQuanYi Zen Hei', 'Microsoft YaHei', sans-serif;
            padding-top: 56px;
        }
        
        #cy {
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        .control-panel {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .node-info-card {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .table-container {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .nav-tabs {
            margin-bottom: 15px;
        }
        
        .layout-btn {
            margin-right: 5px;
            margin-bottom: 5px;
        }
        
        .highlight {
            background-color: yellow !important;
        }
        
        .path-highlight {
            stroke-width: 3px !important;
            stroke-color: #ff0000 !important;
            line-color: #ff0000 !important;
        }
        
        .node-highlight {
            background-color: #ff0000 !important;
        }
        
        .loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }
        
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
        
        .toast-container {
            position: fixed;
            top: 70px;
            right: 20px;
            z-index: 1050;
        }
    </style>
</head>
<body>
    <!-- 导航栏 -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">Streptomyces coelicolor A32 调控网络分析</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item">
                        <a class="nav-link active" href="#" id="network-tab">网络视图</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" id="topology-tab">拓扑分析</a>
                    </li>
                </ul>
                <div class="d-flex">
                    <div class="dropdown">
                        <button class="btn btn-outline-light dropdown-toggle" type="button" id="languageDropdown" data-bs-toggle="dropdown">
                            <i class="fas fa-language"></i> 中文
                        </button>
                        <ul class="dropdown-menu">
                            <li><a class="dropdown-item" href="#" data-lang="zh">中文</a></li>
                            <li><a class="dropdown-item" href="#" data-lang="en">English</a></li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </nav>
    
    <!-- 主内容区 -->
    <div class="container-fluid mt-3">
        <!-- 网络视图 -->
        <div id="network-view" class="view">
            <div class="row">
                <!-- 控制面板 -->
                <div class="col-md-3">
                    <div class="control-panel">
                        <h5>控制面板</h5>
                        
                        <!-- 搜索框 -->
                        <div class="mb-3">
                            <label for="node-search" class="form-label">搜索节点</label>
                            <div class="input-group">
                                <input type="text" class="form-control" id="node-search" placeholder="输入基因ID...">
                                <button class="btn btn-primary" id="search-btn">
                                    <i class="fas fa-search"></i>
                                </button>
                            </div>
                        </div>
                        
                        <!-- 布局选择 -->
                        <div class="mb-3">
                            <label class="form-label">布局算法</label>
                            <div>
                                <button class="btn btn-outline-primary layout-btn" data-layout="cose">CoSE</button>
                                <button class="btn btn-outline-primary layout-btn" data-layout="concentric">Concentric</button>
                                <button class="btn btn-outline-primary layout-btn" data-layout="circle">Circle</button>
                                <button class="btn btn-outline-primary layout-btn" data-layout="breadthfirst">Breadthfirst</button>
                                <button class="btn btn-outline-primary layout-btn" data-layout="random">Random</button>
                            </div>
                        </div>
                        
                        <!-- 邻域深度 -->
                        <div class="mb-3">
                            <label for="neighborhood-depth" class="form-label">邻域深度</label>
                            <select class="form-select" id="neighborhood-depth">
                                <option value="1">1</option>
                                <option value="2">2</option>
                                <option value="3">3</option>
                                <option value="4">4</option>
                                <option value="5">5</option>
                                <option value="6">6</option>
                            </select>
                        </div>
                        
                        <!-- 操作按钮 -->
                        <div class="d-grid gap-2">
                            <button class="btn btn-success" id="expand-btn">
                                <i class="fas fa-expand-alt"></i> 展开邻域
                            </button>
                            <button class="btn btn-info" id="trace-btn">
                                <i class="fas fa-project-diagram"></i> 追溯路径
                            </button>
                            <button class="btn btn-warning" id="shortest-path-btn">
                                <i class="fas fa-route"></i> 最短路径
                            </button>
                            <button class="btn btn-secondary" id="reset-btn">
                                <i class="fas fa-undo"></i> 重置视图
                            </button>
                            <button class="btn btn-danger" id="export-png-btn">
                                <i class="fas fa-image"></i> 导出PNG
                            </button>
                        </div>
                    </div>
                    
                    <!-- 节点信息卡片 -->
                    <div class="card node-info-card">
                        <div class="card-header">
                            <h5 class="mb-0">节点信息</h5>
                        </div>
                        <div class="card-body" id="node-info">
                            <p class="text-muted">请点击网络中的节点查看详细信息</p>
                        </div>
                    </div>
                </div>
                
                <!-- 网络图 -->
                <div class="col-md-9">
                    <div id="cy"></div>
                    
                    <!-- 上下游基因表格 -->
                    <ul class="nav nav-tabs" id="gene-tables-tabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="upstream-tab-btn" data-bs-toggle="tab" data-bs-target="#upstream-tab" type="button" role="tab">
                                上游调控因子
                            </button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="downstream-tab-btn" data-bs-toggle="tab" data-bs-target="#downstream-tab" type="button" role="tab">
                                下游靶基因
                            </button>
                        </li>
                        <li class="nav-item ms-auto">
                            <button class="btn btn-sm btn-success" id="export-csv-btn">
                                <i class="fas fa-file-csv"></i> 导出CSV
                            </button>
                        </li>
                    </ul>
                    <div class="tab-content" id="gene-tables-content">
                        <div class="tab-pane fade show active" id="upstream-tab" role="tabpanel">
                            <div class="table-container mt-2">
                                <table class="table table-striped table-hover" id="upstream-table">
                                    <thead>
                                        <tr>
                                            <th>基因ID</th>
                                            <th>类型</th>
                                            <th>功能</th>
                                            <th>GO Terms</th>
                                            <th>代谢通路</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td colspan="5" class="text-center text-muted">请选择节点查看上游调控因子</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <div class="tab-pane fade" id="downstream-tab" role="tabpanel">
                            <div class="table-container mt-2">
                                <table class="table table-striped table-hover" id="downstream-table">
                                    <thead>
                                        <tr>
                                            <th>基因ID</th>
                                            <th>类型</th>
                                            <th>功能</th>
                                            <th>GO Terms</th>
                                            <th>代谢通路</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td colspan="5" class="text-center text-muted">请选择节点查看下游靶基因</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 拓扑分析视图 -->
        <div id="topology-view" class="view" style="display: none;">
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0">网络拓扑分析</h5>
                        </div>
                        <div class="card-body" id="topology-content">
                            <div class="text-center">
                                <div class="spinner-border" role="status">
                                    <span class="visually-hidden">加载中...</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- 加载提示 -->
    <div class="loading" id="loading" style="display: none;">
        <div class="spinner-border text-primary" role="status">
            <span class="visually-hidden">加载中...</span>
        </div>
    </div>
    
    <!-- Toast 通知 -->
    <div class="toast-container"></div>
    
    <!-- 最短路径对话框 -->
    <div class="modal fade" id="shortest-path-modal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">查找最短路径</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label for="source-node" class="form-label">源节点</label>
                        <input type="text" class="form-control" id="source-node" placeholder="输入源节点ID...">
                    </div>
                    <div class="mb-3">
                        <label for="target-node" class="form-label">目标节点</label>
                        <input type="text" class="form-control" id="target-node" placeholder="输入目标节点ID...">
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                    <button type="button" class="btn btn-primary" id="find-path-btn">查找路径</button>
                </div>
            </div>
        </div>
    </div>
    
    <!-- 自定义JavaScript -->
    <script>
        // 全局变量
        let cy;
        let currentLanguage = 'zh';
        let selectedNode = null;
        let upstreamTable = null;
        let downstreamTable = null;
        
        // 多语言文本
        const texts = {
            zh: {
                networkView: '网络视图',
                topologyView: '拓扑分析',
                searchNode: '搜索节点',
                searchPlaceholder: '输入基因ID...',
                layoutAlgorithm: '布局算法',
                neighborhoodDepth: '邻域深度',
                expandNeighborhood: '展开邻域',
                tracePath: '追溯路径',
                shortestPath: '最短路径',
                resetView: '重置视图',
                exportPNG: '导出PNG',
                nodeInfo: '节点信息',
                clickNodeForInfo: '请点击网络中的节点查看详细信息',
                upstreamRegulators: '上游调控因子',
                downstreamTargets: '下游靶基因',
                exportCSV: '导出CSV',
                selectNodeForUpstream: '请选择节点查看上游调控因子',
                selectNodeForDownstream: '请选择节点查看下游靶基因',
                networkTopology: '网络拓扑分析',
                loading: '加载中...',
                findShortestPath: '查找最短路径',
                sourceNode: '源节点',
                targetNode: '目标节点',
                sourceNodePlaceholder: '输入源节点ID...',
                targetNodePlaceholder: '输入目标节点ID...',
                cancel: '取消',
                findPath: '查找路径',
                noPathFound: '未找到路径',
                geneID: '基因ID',
                type: '类型',
                function: '功能',
                goTerms: 'GO Terms',
                pathways: '代谢通路',
                basicProperties: '基本属性',
                connectivity: '连通性',
                clustering: '聚类系数',
                degreeDistribution: '度分布',
                centrality: '中心性',
                community: '社区结构',
                numNodes: '节点数',
                numEdges: '边数',
                density: '密度',
                isWeaklyConnected: '弱连通',
                isStronglyConnected: '强连通',
                numSCC: '强连通分量数',
                largestSCCSize: '最大强连通分量大小',
                avgPathLength: '平均路径长度',
                avgClustering: '平均聚类系数',
                inDegrees: '入度',
                outDegrees: '出度',
                inDegreeCentrality: '入度中心性',
                outDegreeCentrality: '出度中心性',
                closenessCentrality: '紧密中心性',
                pagerank: 'PageRank',
                numCommunities: '社区数',
                modularity: '模块度',
                success: '成功',
                error: '错误',
                nodeNotFound: '未找到节点',
                networkLoaded: '网络加载成功',
                pathHighlighted: '路径已高亮显示',
                neighborhoodExpanded: '邻域已展开',
                viewReset: '视图已重置',
                imageExported: '图像已导出',
                csvExported: 'CSV文件已导出',
                languageChanged: '语言已切换'
            },
            en: {
                networkView: 'Network View',
                topologyView: 'Topology Analysis',
                searchNode: 'Search Node',
                searchPlaceholder: 'Enter gene ID...',
                layoutAlgorithm: 'Layout Algorithm',
                neighborhoodDepth: 'Neighborhood Depth',
                expandNeighborhood: 'Expand Neighborhood',
                tracePath: 'Trace Path',
                shortestPath: 'Shortest Path',
                resetView: 'Reset View',
                exportPNG: 'Export PNG',
                nodeInfo: 'Node Information',
                clickNodeForInfo: 'Click on a node in the network to view detailed information',
                upstreamRegulators: 'Upstream Regulators',
                downstreamTargets: 'Downstream Targets',
                exportCSV: 'Export CSV',
                selectNodeForUpstream: 'Select a node to view upstream regulators',
                selectNodeForDownstream: 'Select a node to view downstream targets',
                networkTopology: 'Network Topology Analysis',
                loading: 'Loading...',
                findShortestPath: 'Find Shortest Path',
                sourceNode: 'Source Node',
                targetNode: 'Target Node',
                sourceNodePlaceholder: 'Enter source node ID...',
                targetNodePlaceholder: 'Enter target node ID...',
                cancel: 'Cancel',
                findPath: 'Find Path',
                noPathFound: 'No path found',
                geneID: 'Gene ID',
                type: 'Type',
                function: 'Function',
                goTerms: 'GO Terms',
                pathways: 'Pathways',
                basicProperties: 'Basic Properties',
                connectivity: 'Connectivity',
                clustering: 'Clustering',
                degreeDistribution: 'Degree Distribution',
                centrality: 'Centrality',
                community: 'Community Structure',
                numNodes: 'Number of Nodes',
                numEdges: 'Number of Edges',
                density: 'Density',
                isWeaklyConnected: 'Weakly Connected',
                isStronglyConnected: 'Strongly Connected',
                numSCC: 'Number of SCCs',
                largestSCCSize: 'Largest SCC Size',
                avgPathLength: 'Average Path Length',
                avgClustering: 'Average Clustering Coefficient',
                inDegrees: 'In-degrees',
                outDegrees: 'Out-degrees',
                inDegreeCentrality: 'In-degree Centrality',
                outDegreeCentrality: 'Out-degree Centrality',
                closenessCentrality: 'Closeness Centrality',
                pagerank: 'PageRank',
                numCommunities: 'Number of Communities',
                modularity: 'Modularity',
                success: 'Success',
                error: 'Error',
                nodeNotFound: 'Node not found',
                networkLoaded: 'Network loaded successfully',
                pathHighlighted: 'Path highlighted',
                neighborhoodExpanded: 'Neighborhood expanded',
                viewReset: 'View reset',
                imageExported: 'Image exported',
                csvExported: 'CSV file exported',
                languageChanged: 'Language changed'
            }
        };
        
        // 获取当前语言的文本
        function t(key) {
            return texts[currentLanguage][key] || key;
        }
        
        // 显示Toast通知
        function showToast(message, type = 'success') {
            const toastContainer = document.querySelector('.toast-container');
            const toastId = 'toast-' + Date.now();
            
            const toastHtml = `
                <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
                    <div class="toast-header">
                        <i class="fas fa-${type === 'success' ? 'check-circle text-success' : 'exclamation-circle text-danger'} me-2"></i>
                        <strong class="me-auto">${type === 'success' ? t('success') : t('error')}</strong>
                        <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
                    </div>
                    <div class="toast-body">
                        ${message}
                    </div>
                </div>
            `;
            
            toastContainer.insertAdjacentHTML('beforeend', toastHtml);
            
            const toastElement = document.getElementById(toastId);
            const toast = new bootstrap.Toast(toastElement);
            toast.show();
            
            // 自动移除Toast
            toastElement.addEventListener('hidden.bs.toast', function () {
                toastElement.remove();
            });
        }
        
        // 显示/隐藏加载提示
        function toggleLoading(show) {
            document.getElementById('loading').style.display = show ? 'flex' : 'none';
        }
        
        // 初始化网络
        function initNetwork() {
            toggleLoading(true);
            
            // 获取网络数据
            fetch('/api/network')
                .then(response => response.json())
                .then(data => {
                    // 初始化Cytoscape
                    cy = cytoscape({
                        container: document.getElementById('cy'),
                        elements: data.nodes.concat(data.edges),
                        style: [
                            {
                                selector: 'node',
                                style: {
                                    'background-color': '#666',
                                    'label': 'data(id)',
                                    'text-valign': 'center',
                                    'text-halign': 'center',
                                    'font-size': '10px',
                                    'width': 'mapData(total_degree, 0, 10, 20, 50)',
                                    'height': 'mapData(total_degree, 0, 10, 20, 50)'
                                }
                            },
                            {
                                selector: 'node[type="regulator"]',
                                style: {
                                    'background-color': '#e74c3c'
                                }
                            },
                            {
                                selector: 'node[type="target"]',
                                style: {
                                    'background-color': '#3498db'
                                }
                            },
                            {
                                selector: 'node:selected',
                                style: {
                                    'border-width': '3px',
                                    'border-color': '#f39c12',
                                    'background-color': '#f39c12'
                                }
                            },
                            {
                                selector: 'node.highlight',
                                style: {
                                    'background-color': '#f1c40f'
                                }
                            },
                            {
                                selector: 'node.path-highlight',
                                style: {
                                    'background-color': '#e74c3c',
                                    'border-width': '3px',
                                    'border-color': '#c0392b'
                                }
                            },
                            {
                                selector: 'edge',
                                style: {
                                    'width': 1,
                                    'line-color': '#999',
                                    'target-arrow-color': '#999',
                                    'target-arrow-shape': 'triangle',
                                    'curve-style': 'bezier'
                                }
                            },
                            {
                                selector: 'edge[reg_to_reg=true]',
                                style: {
                                    'line-color': '#e74c3c',
                                    'target-arrow-color': '#e74c3c',
                                    'width': 2
                                }
                            },
                            {
                                selector: 'edge.highlight',
                                style: {
                                    'line-color': '#f1c40f',
                                    'target-arrow-color': '#f1c40f',
                                    'width': 3
                                }
                            },
                            {
                                selector: 'edge.path-highlight',
                                style: {
                                    'line-color': '#e74c3c',
                                    'target-arrow-color': '#e74c3c',
                                    'width': 3
                                }
                            }
                        ],
                        layout: {
                            name: 'cose',
                            animate: true,
                            animationDuration: 1000,
                            nodeRepulsion: 100000,
                            idealEdgeLength: 100,
                            edgeElasticity: 100,
                            nestingFactor: 5,
                            gravity: 80,
                            numIter: 1000,
                            initialTemp: 200,
                            coolingFactor: 0.95,
                            minTemp: 1.0
                        }
                    });
                    
                    // 添加节点点击事件
                    cy.on('tap', 'node', function(evt) {
                        const node = evt.target;
                        selectNode(node.id());
                    });
                    
                    // 添加背景点击事件
                    cy.on('tap', function(evt) {
                        if (evt.target === cy) {
                            deselectNode();
                        }
                    });
                    
                    // 初始化表格
                    initTables();
                    
                    toggleLoading(false);
                    showToast(t('networkLoaded'));
                })
                .catch(error => {
                    console.error('Error loading network:', error);
                    toggleLoading(false);
                    showToast(t('error') + ': ' + error.message, 'error');
                });
        }
        
        // 初始化表格
        function initTables() {
            // 上游调控因子表格
            upstreamTable = $('#upstream-table').DataTable({
                language: {
                    url: currentLanguage === 'zh' ? '//cdn.datatables.net/plug-ins/1.11.5/i18n/zh.json' : '//cdn.datatables.net/plug-ins/1.11.5/i18n/en.json'
                },
                pageLength: 10,
                responsive: true
            });
            
            // 下游靶基因表格
            downstreamTable = $('#downstream-table').DataTable({
                language: {
                    url: currentLanguage === 'zh' ? '//cdn.datatables.net/plug-ins/1.11.5/i18n/zh.json' : '//cdn.datatables.net/plug-ins/1.11.5/i18n/en.json'
                },
                pageLength: 10,
                responsive: true
            });
        }
        
        // 选择节点
        function selectNode(nodeId) {
            // 高亮节点
            cy.nodes().removeClass('highlight');
            cy.getElementById(nodeId).addClass('highlight');
            
            // 获取节点信息
            toggleLoading(true);
            
            fetch(`/api/node/${nodeId}`)
                .then(response => response.json())
                .then(data => {
                    selectedNode = nodeId;
                    
                    // 显示节点信息
                    displayNodeInfo(data);
                    
                    // 更新表格
                    updateTables(nodeId);
                    
                    toggleLoading(false);
                })
                .catch(error => {
                    console.error('Error getting node info:', error);
                    toggleLoading(false);
                    showToast(t('error') + ': ' + error.message, 'error');
                });
        }
        
        // 取消选择节点
        function deselectNode() {
            selectedNode = null;
            cy.nodes().removeClass('highlight');
            
            // 清空节点信息
            document.getElementById('node-info').innerHTML = `<p class="text-muted">${t('clickNodeForInfo')}</p>`;
            
            // 清空表格
            upstreamTable.clear().draw();
            downstreamTable.clear().draw();
        }
        
        // 显示节点信息
        function displayNodeInfo(nodeInfo) {
            if (!nodeInfo || Object.keys(nodeInfo).length === 0) {
                document.getElementById('node-info').innerHTML = `<p class="text-muted">${t('nodeNotFound')}</p>`;
                return;
            }
            
            let html = `
                <h6>${nodeInfo.id}</h6>
                <p><strong>${t('type')}:</strong> ${nodeInfo.type}</p>
                <p><strong>${t('inDegrees')}:</strong> ${nodeInfo.in_degree}</p>
                <p><strong>${t('outDegrees')}:</strong> ${nodeInfo.out_degree}</p>
                <p><strong>${t('totalDegree')}:</strong> ${nodeInfo.total_degree}</p>
                <p><strong>SCC ID:</strong> ${nodeInfo.scc_id}</p>
            `;
            
            if (nodeInfo.function) {
                html += `<p><strong>${t('function')}:</strong> ${nodeInfo.function}</p>`;
            }
            
            if (nodeInfo.go_terms && nodeInfo.go_terms.length > 0) {
                html += `<p><strong>${t('goTerms')}:</strong> ${nodeInfo.go_terms.join(', ')}</p>`;
            }
            
            if (nodeInfo.pathways && nodeInfo.pathways.length > 0) {
                html += `<p><strong>${t('pathways')}:</strong> ${nodeInfo.pathways.join(', ')}</p>`;
            }
            
            document.getElementById('node-info').innerHTML = html;
        }
        
        // 更新表格
        function updateTables(nodeId) {
            toggleLoading(true);
            
            fetch(`/api/table/${nodeId}`)
                .then(response => response.json())
                .then(data => {
                    // 更新上游调控因子表格
                    upstreamTable.clear();
                    if (data.upstream && data.upstream.length > 0) {
                        data.upstream.forEach(item => {
                            upstreamTable.row.add([
                                item.id,
                                item.type,
                                item.function || '',
                                item.go_terms || '',
                                item.pathways || ''
                            ]);
                        });
                    } else {
                        upstreamTable.row.add([
                            `<td colspan="5" class="text-center text-muted">${t('selectNodeForUpstream')}</td>`
                        ]);
                    }
                    upstreamTable.draw();
                    
                    // 更新下游靶基因表格
                    downstreamTable.clear();
                    if (data.downstream && data.downstream.length > 0) {
                        data.downstream.forEach(item => {
                            downstreamTable.row.add([
                                item.id,
                                item.type,
                                item.function || '',
                                item.go_terms || '',
                                item.pathways || ''
                            ]);
                        });
                    } else {
                        downstreamTable.row.add([
                            `<td colspan="5" class="text-center text-muted">${t('selectNodeForDownstream')}</td>`
                        ]);
                    }
                    downstreamTable.draw();
                    
                    toggleLoading(false);
                })
                .catch(error => {
                    console.error('Error getting table data:', error);
                    toggleLoading(false);
                    showToast(t('error') + ': ' + error.message, 'error');
                });
        }
        
        // 展开邻域
        function expandNeighborhood() {
            if (!selectedNode) {
                showToast(t('nodeNotFound'), 'error');
                return;
            }
            
            const depth = parseInt(document.getElementById('neighborhood-depth').value);
            
            toggleLoading(true);
            
            fetch('/api/subnetwork', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    nodes: [selectedNode],
                    depth: depth
                })
            })
            .then(response => response.json())
            .then(data => {
                // 清除现有高亮
                cy.elements().removeClass('highlight');
                
                // 添加新元素
                const newElements = [];
                
                // 添加节点
                data.nodes.forEach(node => {
                    if (!cy.getElementById(node.data.id).length) {
                        newElements.push(node);
                    } else {
                        // 高亮现有节点
                        cy.getElementById(node.data.id).addClass('highlight');
                    }
                });
                
                // 添加边
                data.edges.forEach(edge => {
                    if (!cy.getElementById(edge.data.id).length) {
                        newElements.push(edge);
                    } else {
                        // 高亮现有边
                        cy.getElementById(edge.data.id).addClass('highlight');
                    }
                });
                
                // 添加新元素到网络
                if (newElements.length > 0) {
                    cy.add(newElements);
                }
                
                // 应用布局
                cy.layout({
                    name: 'cose',
                    animate: true,
                    animationDuration: 1000,
                    nodeRepulsion: 100000,
                    idealEdgeLength: 100,
                    edgeElasticity: 100,
                    nestingFactor: 5,
                    gravity: 80,
                    numIter: 1000,
                    initialTemp: 200,
                    coolingFactor: 0.95,
                    minTemp: 1.0
                }).run();
                
                toggleLoading(false);
                showToast(t('neighborhoodExpanded'));
            })
            .catch(error => {
                console.error('Error expanding neighborhood:', error);
                toggleLoading(false);
                showToast(t('error') + ': ' + error.message, 'error');
            });
        }
        
        // 追溯路径
        function tracePath() {
            if (!selectedNode) {
                showToast(t('nodeNotFound'), 'error');
                return;
            }
            
            const depth = parseInt(document.getElementById('neighborhood-depth').value);
            
            toggleLoading(true);
            
            fetch('/api/subnetwork', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    nodes: [selectedNode],
                    depth: depth
                })
            })
            .then(response => response.json())
            .then(data => {
                // 清除现有高亮
                cy.elements().removeClass('highlight');
                
                // 添加新元素
                const newElements = [];
                
                // 添加节点
                data.nodes.forEach(node => {
                    if (!cy.getElementById(node.data.id).length) {
                        newElements.push(node);
                    } else {
                        // 高亮现有节点
                        cy.getElementById(node.data.id).addClass('highlight');
                    }
                });
                
                // 添加边
                data.edges.forEach(edge => {
                    if (!cy.getElementById(edge.data.id).length) {
                        newElements.push(edge);
                    } else {
                        // 高亮现有边
                        cy.getElementById(edge.data.id).addClass('highlight');
                    }
                });
                
                // 添加新元素到网络
                if (newElements.length > 0) {
                    cy.add(newElements);
                }
                
                // 应用布局
                cy.layout({
                    name: 'breadthfirst',
                    directed: true,
                    animate: true,
                    animationDuration: 1000,
                    roots: `#${selectedNode}`
                }).run();
                
                toggleLoading(false);
                showToast(t('pathHighlighted'));
            })
            .catch(error => {
                console.error('Error tracing path:', error);
                toggleLoading(false);
                showToast(t('error') + ': ' + error.message, 'error');
            });
        }
        
        // 查找最短路径
        function findShortestPath(source, target) {
            toggleLoading(true);
            
            fetch('/api/shortest_path', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    source: source,
                    target: target
                })
            })
            .then(response => response.json())
            .then(data => {
                // 清除现有高亮
                cy.elements().removeClass('path-highlight');
                
                if (data.path && data.path.length > 0) {
                    // 高亮路径节点
                    data.path.forEach(nodeId => {
                        cy.getElementById(nodeId).addClass('path-highlight');
                    });
                    
                    // 高亮路径边
                    for (let i = 0; i < data.path.length - 1; i++) {
                        const sourceId = data.path[i];
                        const targetId = data.path[i + 1];
                        const edge = cy.edges(`[source="${sourceId}"][target="${targetId}"]`);
                        if (edge.length > 0) {
                            edge.addClass('path-highlight');
                        }
                    }
                    
                    // 居中显示路径
                    const pathNodes = data.path.map(nodeId => cy.getElementById(nodeId));
                    cy.center(pathNodes);
                    
                    showToast(t('pathHighlighted'));
                } else {
                    showToast(t('noPathFound'), 'error');
                }
                
                toggleLoading(false);
            })
            .catch(error => {
                console.error('Error finding shortest path:', error);
                toggleLoading(false);
                showToast(t('error') + ': ' + error.message, 'error');
            });
        }
        
        // 重置视图
        function resetView() {
            // 清除高亮
            cy.elements().removeClass('highlight path-highlight');
            
            // 重新应用布局
            cy.layout({
                name: 'cose',
                animate: true,
                animationDuration: 1000,
                nodeRepulsion: 100000,
                idealEdgeLength: 100,
                edgeElasticity: 100,
                nestingFactor: 5,
                gravity: 80,
                numIter: 1000,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0
            }).run();
            
            // 取消选择节点
            deselectNode();
            
            showToast(t('viewReset'));
        }
        
        // 导出PNG
        function exportPNG() {
            toggleLoading(true);
            
            const layout = cy.scratch('currentLayoutName') || 'cose';
            
            fetch('/api/export/png', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    layout: layout
                })
            })
            .then(response => response.json())
            .then(data => {
                // 创建下载链接
                const link = document.createElement('a');
                link.href = 'data:image/png;base64,' + data.image;
                link.download = data.filename;
                link.click();
                
                toggleLoading(false);
                showToast(t('imageExported'));
            })
            .catch(error => {
                console.error('Error exporting PNG:', error);
                toggleLoading(false);
                showToast(t('error') + ': ' + error.message, 'error');
            });
        }
        
        // 导出CSV
        function exportCSV() {
            if (!selectedNode) {
                showToast(t('nodeNotFound'), 'error');
                return;
            }
            
            const activeTab = document.querySelector('#gene-tables-tabs .nav-link.active').id;
            const tableType = activeTab === 'upstream-tab-btn' ? 'upstream' : 'downstream';
            
            toggleLoading(true);
            
            // 获取表格数据
            const table = tableType === 'upstream' ? upstreamTable : downstreamTable;
            const data = [];
            
            table.rows().every(function() {
                const row = this.data();
                if (row && row.length > 0 && !row[0].includes('colspan')) {
                    data.push({
                        'Gene ID': row[0],
                        'Type': row[1],
                        'Function': row[2],
                        'GO Terms': row[3],
                        'Pathways': row[4]
                    });
                }
            });
            
            fetch('/api/export/csv', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    data: data,
                    type: tableType,
                    node_id: selectedNode
                })
            })
            .then(response => response.json())
            .then(data => {
                // 创建下载链接
                const link = document.createElement('a');
                link.href = 'data:text/csv;base64,' + data.file;
                link.download = data.filename;
                link.click();
                
                toggleLoading(false);
                showToast(t('csvExported'));
            })
            .catch(error => {
                console.error('Error exporting CSV:', error);
                toggleLoading(false);
                showToast(t('error') + ': ' + error.message, 'error');
            });
        }
        
        // 加载拓扑分析数据
        function loadTopologyAnalysis() {
            toggleLoading(true);
            
            fetch('/api/topology')
                .then(response => response.json())
                .then(data => {
                    let html = '';
                    
                    // 基本属性
                    html += `
                        <div class="row mb-4">
                            <div class="col-md-12">
                                <h5>${t('basicProperties')}</h5>
                                <div class="table-responsive">
                                    <table class="table table-striped">
                                        <tbody>
                                            <tr>
                                                <th>${t('numNodes')}</th>
                                                <td>${data.basic_properties.num_nodes}</td>
                                            </tr>
                                            <tr>
                                                <th>${t('numEdges')}</th>
                                                <td>${data.basic_properties.num_edges}</td>
                                            </tr>
                                            <tr>
                                                <th>${t('density')}</th>
                                                <td>${data.basic_properties.density.toFixed(4)}</td>
                                            </tr>
                                            <tr>
                                                <th>${t('isWeaklyConnected')}</th>
                                                <td>${data.basic_properties.is_weakly_connected ? 'Yes' : 'No'}</td>
                                            </tr>
                                            <tr>
                                                <th>${t('isStronglyConnected')}</th>
                                                <td>${data.basic_properties.is_strongly_connected ? 'Yes' : 'No'}</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // 连通性
                    html += `
                        <div class="row mb-4">
                            <div class="col-md-12">
                                <h5>${t('connectivity')}</h5>
                                <div class="table-responsive">
                                    <table class="table table-striped">
                                        <tbody>
                                            <tr>
                                                <th>${t('numSCC')}</th>
                                                <td>${data.connectivity.num_scc}</td>
                                            </tr>
                                            <tr>
                                                <th>${t('largestSCCSize')}</th>
                                                <td>${data.connectivity.largest_scc_size}</td>
                                            </tr>
                `;
                    
                    if (data.connectivity.avg_path_length !== null) {
                        html += `
                                            <tr>
                                                <th>${t('avgPathLength')}</th>
                                                <td>${data.connectivity.avg_path_length.toFixed(4)}</td>
                                            </tr>
                        `;
                    }
                    
                    html += `
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // 聚类系数
                    html += `
                        <div class="row mb-4">
                            <div class="col-md-12">
                                <h5>${t('clustering')}</h5>
                                <div class="table-responsive">
                                    <table class="table table-striped">
                                        <tbody>
                                            <tr>
                                                <th>${t('avgClustering')}</th>
                                                <td>${data.clustering.avg_clustering.toFixed(4)}</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // 度分布
                    html += `
                        <div class="row mb-4">
                            <div class="col-md-6">
                                <h5>${t('inDegrees')}</h5>
                                <div class="table-responsive">
                                    <table class="table table-striped">
                                        <thead>
                                            <tr>
                                                <th>度数</th>
                                                <th>节点数</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                    `;
                    
                    // 计算入度分布
                    const inDegreeDist = {};
                    data.degree_distribution.in_degrees.forEach(degree => {
                        inDegreeDist[degree] = (inDegreeDist[degree] || 0) + 1;
                    });
                    
                    Object.keys(inDegreeDist).sort((a, b) => a - b).forEach(degree => {
                        html += `
                                            <tr>
                                                <td>${degree}</td>
                                                <td>${inDegreeDist[degree]}</td>
                                            </tr>
                        `;
                    });
                    
                    html += `
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <h5>${t('outDegrees')}</h5>
                                <div class="table-responsive">
                                    <table class="table table-striped">
                                        <thead>
                                            <tr>
                                                <th>度数</th>
                                                <th>节点数</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                    `;
                    
                    // 计算出度分布
                    const outDegreeDist = {};
                    data.degree_distribution.out_degrees.forEach(degree => {
                        outDegreeDist[degree] = (outDegreeDist[degree] || 0) + 1;
                    });
                    
                    Object.keys(outDegreeDist).sort((a, b) => a - b).forEach(degree => {
                        html += `
                                            <tr>
                                                <td>${degree}</td>
                                                <td>${outDegreeDist[degree]}</td>
                                            </tr>
                        `;
                    });
                    
                    html += `
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // 中心性
                    if (data.centrality.in_degree_centrality) {
                        html += `
                            <div class="row mb-4">
                                <div class="col-md-12">
                                    <h5>${t('centrality')}</h5>
                                    <div class="table-responsive">
                                        <table class="table table-striped">
                                            <thead>
                                                <tr>
                                                    <th>节点</th>
                                                    <th>${t('inDegreeCentrality')}</th>
                                                    <th>${t('outDegreeCentrality')}</th>
                        `;
                        
                        if (data.centrality.closeness_centrality) {
                            html += `<th>${t('closenessCentrality')}</th>`;
                        }
                        
                        html += `<th>${t('pagerank')}</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                        `;
                        
                        // 获取所有节点
                        const nodes = Object.keys(data.centrality.in_degree_centrality);
                        
                        // 按PageRank排序
                        nodes.sort((a, b) => {
                            return data.centrality.pagerank[b] - data.centrality.pagerank[a];
                        });
                        
                        // 只显示前20个节点
                        const topNodes = nodes.slice(0, 20);
                        
                        topNodes.forEach(node => {
                            html += `
                                                <tr>
                                                    <td>${node}</td>
                                                    <td>${data.centrality.in_degree_centrality[node].toFixed(4)}</td>
                                                    <td>${data.centrality.out_degree_centrality[node].toFixed(4)}</td>
                            `;
                            
                            if (data.centrality.closeness_centrality) {
                                html += `<td>${data.centrality.closeness_centrality[node].toFixed(4)}</td>`;
                            }
                            
                            html += `<td>${data.centrality.pagerank[node].toFixed(4)}</td>
                                                </tr>
                            `;
                        });
                        
                        html += `
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    
                    // 社区结构
                    if (data.community.communities) {
                        html += `
                            <div class="row mb-4">
                                <div class="col-md-12">
                                    <h5>${t('community')}</h5>
                                    <div class="table-responsive">
                                        <table class="table table-striped">
                                            <tbody>
                                                <tr>
                                                    <th>${t('numCommunities')}</th>
                                                    <td>${data.community.num_communities}</td>
                                                </tr>
                                                <tr>
                                                    <th>${t('modularity')}</th>
                                                    <td>${data.community.modularity.toFixed(4)}</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    
                    document.getElementById('topology-content').innerHTML = html;
                    toggleLoading(false);
                })
                .catch(error => {
                    console.error('Error loading topology analysis:', error);
                    toggleLoading(false);
                    showToast(t('error') + ': ' + error.message, 'error');
                });
        }
        
        // 更新界面语言
        function updateLanguage(lang) {
            currentLanguage = lang;
            
            // 更新导航栏
            document.querySelector('#network-tab').textContent = t('networkView');
            document.querySelector('#topology-tab').textContent = t('topologyView');
            document.querySelector('#languageDropdown').innerHTML = `<i class="fas fa-language"></i> ${lang === 'zh' ? '中文' : 'English'}`;
            
            // 更新控制面板
            document.querySelector('label[for="node-search"]').textContent = t('searchNode');
            document.querySelector('#node-search').placeholder = t('searchPlaceholder');
            document.querySelector('label[for="neighborhood-depth"]').textContent = t('neighborhoodDepth');
            document.querySelector('#expand-btn').innerHTML = `<i class="fas fa-expand-alt"></i> ${t('expandNeighborhood')}`;
            document.querySelector('#trace-btn').innerHTML = `<i class="fas fa-project-diagram"></i> ${t('tracePath')}`;
            document.querySelector('#shortest-path-btn').innerHTML = `<i class="fas fa-route"></i> ${t('shortestPath')}`;
            document.querySelector('#reset-btn').innerHTML = `<i class="fas fa-undo"></i> ${t('resetView')}`;
            document.querySelector('#export-png-btn').innerHTML = `<i class="fas fa-image"></i> ${t('exportPNG')}`;
            
            // 更新节点信息卡片
            document.querySelector('.node-info-card .card-header h5').textContent = t('nodeInfo');
            if (!selectedNode) {
                document.getElementById('node-info').innerHTML = `<p class="text-muted">${t('clickNodeForInfo')}</p>`;
            }
            
            // 更新表格标签
            document.querySelector('#upstream-tab-btn').textContent = t('upstreamRegulators');
            document.querySelector('#downstream-tab-btn').textContent = t('downstreamTargets');
            document.querySelector('#export-csv-btn').innerHTML = `<i class="fas fa-file-csv"></i> ${t('exportCSV')}`;
            
            // 更新表格列标题
            if (upstreamTable) {
                upstreamTable.column(0).header().textContent = t('geneID');
                upstreamTable.column(1).header().textContent = t('type');
                upstreamTable.column(2).header().textContent = t('function');
                upstreamTable.column(3).header().textContent = t('goTerms');
                upstreamTable.column(4).header().textContent = t('pathways');
            }
            
            if (downstreamTable) {
                downstreamTable.column(0).header().textContent = t('geneID');
                downstreamTable.column(1).header().textContent = t('type');
                downstreamTable.column(2).header().textContent = t('function');
                downstreamTable.column(3).header().textContent = t('goTerms');
                downstreamTable.column(4).header().textContent = t('pathways');
            }
            
            // 更新最短路径对话框
            document.querySelector('#shortest-path-modal .modal-title').textContent = t('findShortestPath');
            document.querySelector('label[for="source-node"]').textContent = t('sourceNode');
            document.querySelector('#source-node').placeholder = t('sourceNodePlaceholder');
            document.querySelector('label[for="target-node"]').textContent = t('targetNode');
            document.querySelector('#target-node').placeholder = t('targetNodePlaceholder');
            document.querySelector('#find-path-btn').textContent = t('findPath');
            document.querySelector('#shortest-path-modal .btn-secondary').textContent = t('cancel');
            
            // 更新拓扑分析视图
            document.querySelector('#topology-view .card-header h5').textContent = t('networkTopology');
            
            // 重新加载拓扑分析数据（如果当前在拓扑分析视图）
            if (document.getElementById('topology-view').style.display !== 'none') {
                loadTopologyAnalysis();
            }
            
            showToast(t('languageChanged'));
        }
        
        // 初始化事件监听器
        function initEventListeners() {
            // 标签页切换
            document.getElementById('network-tab').addEventListener('click', function(e) {
                e.preventDefault();
                document.getElementById('network-view').style.display = 'block';
                document.getElementById('topology-view').style.display = 'none';
                document.getElementById('network-tab').classList.add('active');
                document.getElementById('topology-tab').classList.remove('active');
            });
            
            document.getElementById('topology-tab').addEventListener('click', function(e) {
                e.preventDefault();
                document.getElementById('network-view').style.display = 'none';
                document.getElementById('topology-view').style.display = 'block';
                document.getElementById('network-tab').classList.remove('active');
                document.getElementById('topology-tab').classList.add('active');
                loadTopologyAnalysis();
            });
            
            // 搜索节点
            document.getElementById('search-btn').addEventListener('click', function() {
                const searchTerm = document.getElementById('node-search').value.trim();
                if (searchTerm) {
                    const node = cy.getElementById(searchTerm);
                    if (node.length > 0) {
                        selectNode(searchTerm);
                        cy.center(node);
                    } else {
                        showToast(t('nodeNotFound'), 'error');
                    }
                }
            });
            
            // 布局选择
            document.querySelectorAll('.layout-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const layout = this.getAttribute('data-layout');
                    
                    // 应用布局
                    let layoutOptions;
                    
                    switch (layout) {
                        case 'cose':
                            layoutOptions = {
                                name: 'cose',
                                animate: true,
                                animationDuration: 1000,
                                nodeRepulsion: 100000,
                                idealEdgeLength: 100,
                                edgeElasticity: 100,
                                nestingFactor: 5,
                                gravity: 80,
                                numIter: 1000,
                                initialTemp: 200,
                                coolingFactor: 0.95,
                                minTemp: 1.0
                            };
                            break;
                        case 'concentric':
                            layoutOptions = {
                                name: 'concentric',
                                animate: true,
                                animationDuration: 1000,
                                concentric: function(node) {
                                    return node.data('total_degree');
                                },
                                levelWidth: function(nodes) {
                                    return 2;
                                },
                                minNodeSpacing: 10
                            };
                            break;
                        case 'circle':
                            layoutOptions = {
                                name: 'circle',
                                animate: true,
                                animationDuration: 1000,
                                radius: Math.min(cy.width(), cy.height()) / 2 - 50
                            };
                            break;
                        case 'breadthfirst':
                            layoutOptions = {
                                name: 'breadthfirst',
                                directed: true,
                                animate: true,
                                animationDuration: 1000,
                                roots: selectedNode ? `#${selectedNode}` : undefined
                            };
                            break;
                        case 'random':
                            layoutOptions = {
                                name: 'random',
                                animate: true,
                                animationDuration: 1000
                            };
                            break;
                        default:
                            layoutOptions = {
                                name: 'cose',
                                animate: true,
                                animationDuration: 1000
                            };
                    }
                    
                    cy.layout(layoutOptions).run();
                    cy.scratch('currentLayoutName', layout);
                });
            });
            
            // 展开邻域
            document.getElementById('expand-btn').addEventListener('click', expandNeighborhood);
            
            // 追溯路径
            document.getElementById('trace-btn').addEventListener('click', tracePath);
            
            // 最短路径
            document.getElementById('shortest-path-btn').addEventListener('click', function() {
                const modal = new bootstrap.Modal(document.getElementById('shortest-path-modal'));
                modal.show();
            });
            
            document.getElementById('find-path-btn').addEventListener('click', function() {
                const source = document.getElementById('source-node').value.trim();
                const target = document.getElementById('target-node').value.trim();
                
                if (source && target) {
                    findShortestPath(source, target);
                    bootstrap.Modal.getInstance(document.getElementById('shortest-path-modal')).hide();
                }
            });
            
            // 重置视图
            document.getElementById('reset-btn').addEventListener('click', resetView);
            
            // 导出PNG
            document.getElementById('export-png-btn').addEventListener('click', exportPNG);
            
            // 导出CSV
            document.getElementById('export-csv-btn').addEventListener('click', exportCSV);
            
            // 语言切换
            document.querySelectorAll('[data-lang]').forEach(item => {
                item.addEventListener('click', function(e) {
                    e.preventDefault();
                    const lang = this.getAttribute('data-lang');
                    
                    // 更新服务器端语言设置
                    fetch(`/api/language/${lang}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                updateLanguage(lang);
                            }
                        })
                        .catch(error => {
                            console.error('Error changing language:', error);
                            showToast(t('error') + ': ' + error.message, 'error');
                        });
                });
            });
        }
        
        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', function() {
            initEventListeners();
            initNetwork();
        });
    </script>
</body>
</html>
    """
    
    with open(os.path.join(template_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html)


def main():
    """主函数，用于运行应用"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Streptomyces coelicolor A32 调控网络分析应用')
    parser.add_argument('--data', type=str, help='调控网络数据文件路径 (TSV/CSV)')
    parser.add_argument('--go', type=str, help='GO Terms注释文件路径')
    parser.add_argument('--pathway', type=str, help='代谢通路注释文件路径')
    parser.add_argument('--function', type=str, help='基因功能注释文件路径')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    
    args = parser.parse_args()
    
    # 创建HTML模板
    create_html_template()
    
    # 创建并运行应用
    app = RegulatoryNetworkApp(
        data_file=args.data,
        go_file=args.go,
        pathway_file=args.pathway,
        function_file=args.function
    )
    
    print("启动 Streptomyces coelicolor A32 调控网络分析应用...")
    print(f"访问地址: http://{args.host}:{args.port}")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        open_browser=not args.no_browser
    )


if __name__ == '__main__':
    main()