import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import os

class EnhancedRGWLKernel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.get("max_features", 200),
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        
        self.n_clusters = config.get("n_clusters", 10)
        self.kmeans = None
        
        
        self.node_feature_cache = {}
        self.label_cache = {}
        self.subgraph_cache = {}
        
        
        self.computation_stats = {
            "kernel_computations": 0,
            "subgraph_extractions": 0,
            "label_propagations": 0,
            "cache_hits": 0
        }
        
        
        self._quiet = os.environ.get('BENCHMARK_QUIET') == '1'

        def _qprint(*args, **kwargs):
            if not self._quiet:
                print(*args, **kwargs)
        self._qprint = _qprint
        
        self._qprint("[🧮] Enhanced RGWL kernel initialized")
    
    def compute_kernel(self, 
                      G1: Dict[str, Any], 
                      G2: Dict[str, Any], 
                      h: int = 3,
                      save_intermediate: bool = False) -> Dict[str, Any]:
        self._qprint(f"[🧮] Start computing RGWL kernel, iterations: {h}")
        
        self.computation_stats["kernel_computations"] += 1
        
        
        lambda1 = G1.get("metadata", {}).get("lambda_ratio", 0.0)
        lambda2 = G2.get("metadata", {}).get("lambda_ratio", 0.0)
        self._qprint(f"[📊] λ1 = {lambda1:.4f}, λ2 = {lambda2:.4f}")
        
        
        intermediate_results = {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "round_contributions": [],
            "answer_subgraphs": {},
            "full_graphs": {},
            "label_propagation_history": [],
            "histogram_evolution": []
        }
        
        self._qprint("[🔍] Step 1: extract answer-contribution subgraph")
        answer_subgraph1 = self._extract_answer_subgraph(G1, "G1")
        answer_subgraph2 = self._extract_answer_subgraph(G2, "G2")
        
        intermediate_results["answer_subgraphs"] = {
            "G1": answer_subgraph1,
            "G2": answer_subgraph2
        }
        
        self._qprint("[🏷️] Step 2: node labeling")
        labeled_answer_graph1 = self._label_nodes_knn(answer_subgraph1, "answer_G1")
        labeled_answer_graph2 = self._label_nodes_knn(answer_subgraph2, "answer_G2")
        
        labeled_full_graph1 = self._label_nodes_knn(G1, "full_G1")
        labeled_full_graph2 = self._label_nodes_knn(G2, "full_G2")
        
        intermediate_results["full_graphs"] = {
            "labeled_answer_G1": labeled_answer_graph1,
            "labeled_answer_G2": labeled_answer_graph2,
            "labeled_full_G1": labeled_full_graph1,
            "labeled_full_G2": labeled_full_graph2
        }
        
        
        kernel_sum = 0.0
        answer_term_sum = 0.0
        full_term_sum = 0.0
        
        for i in range(1, h + 1):
            self._qprint(f"[🔄] Steps 3-5: round {i} propagation and histograms")
            
            round_data = {
                "round": i,
                "propagations": {},
                "histograms": {},
                "inner_products": {}
            }
            
            
            propagated_answer1 = self._directed_label_propagation(labeled_answer_graph1, i, f"answer_G1_r{i}")
            propagated_answer2 = self._directed_label_propagation(labeled_answer_graph2, i, f"answer_G2_r{i}")
            
            propagated_full1 = self._directed_label_propagation(labeled_full_graph1, i, f"full_G1_r{i}")
            propagated_full2 = self._directed_label_propagation(labeled_full_graph2, i, f"full_G2_r{i}")
            
            round_data["propagations"] = {
                "answer_G1": propagated_answer1,
                "answer_G2": propagated_answer2,
                "full_G1": propagated_full1,
                "full_G2": propagated_full2
            }
            
            
            hist_answer1 = self._compute_label_histogram(propagated_answer1, f"hist_answer_G1_r{i}")
            hist_answer2 = self._compute_label_histogram(propagated_answer2, f"hist_answer_G2_r{i}")
            
            hist_full1 = self._compute_label_histogram(propagated_full1, f"hist_full_G1_r{i}")
            hist_full2 = self._compute_label_histogram(propagated_full2, f"hist_full_G2_r{i}")
            
            round_data["histograms"] = {
                "answer_G1": hist_answer1,
                "answer_G2": hist_answer2,
                "full_G1": hist_full1,
                "full_G2": hist_full2
            }
            
            
            inner_product_answer = self._compute_inner_product(hist_answer1, hist_answer2)
            inner_product_full = self._compute_inner_product(hist_full1, hist_full2)
            
            round_data["inner_products"] = {
                "answer_term": inner_product_answer,
                "full_term": inner_product_full
            }
            
            
            answer_contribution = inner_product_answer
            full_contribution = lambda1 * lambda2 * inner_product_full
            round_contribution = answer_contribution + full_contribution
            
            answer_term_sum += answer_contribution
            full_term_sum += full_contribution
            kernel_sum += round_contribution
            
            round_data.update({
                "answer_contribution": answer_contribution,
                "full_contribution": full_contribution,
                "round_contribution": round_contribution,
                "cumulative_kernel": kernel_sum
            })
            
            intermediate_results["round_contributions"].append(round_data)
            
            self._qprint(f"[📊] Round {i}:")
            self._qprint(f"    Answer term: {answer_contribution:.6f}")
            self._qprint(f"    Full term: {full_contribution:.6f}")
            self._qprint(f"    Round contribution: {round_contribution:.6f}")
            self._qprint(f"    Cumulative kernel: {kernel_sum:.6f}")
        
        
        final_result = {
            "kernel_value": kernel_sum,
            "answer_term_sum": answer_term_sum,
            "full_term_sum": full_term_sum,
            "lambda_product": lambda1 * lambda2,
            "intermediate_results": intermediate_results if save_intermediate else None,
            "computation_stats": self.computation_stats.copy(),
            "graph_properties": {
                "G1": self._analyze_graph_properties(G1),
                "G2": self._analyze_graph_properties(G2)
            }
        }
        
        self._qprint(f"[✅] RGWL kernel completed:")
        self._qprint(f"    Final kernel value: {kernel_sum:.6f}")
        self._qprint(f"    Answer term sum: {answer_term_sum:.6f}")
        self._qprint(f"    Full term sum: {full_term_sum:.6f}")
        self._qprint(f"    Lambda product: {lambda1 * lambda2:.6f}")
        
        return final_result

    def validate_kernel_matrix(self, 
                               kernel_matrix: np.ndarray,
                               report_path: Optional[str] = None,
                               title: str = "RGWL kernel matrix WLTest report",
                               atol: float = 1e-8) -> Dict[str, Any]:
        try:
            K = np.asarray(kernel_matrix, dtype=float)
            n, m = K.shape
            sym = bool(np.allclose(K, K.T, atol=atol))
            evals = np.linalg.eigvalsh((K + K.T) / 2.0)
            min_eig = float(np.min(evals)) if evals.size else 0.0
            neg_count = int(np.sum(evals < -atol))
            psd = bool(min_eig >= -atol)
            trace = float(np.trace(K))
            frob = float(np.linalg.norm(K, ord='fro'))
            max_abs = float(np.max(np.abs(evals))) if evals.size else 0.0
            min_pos = float(np.min(evals[evals > atol])) if np.any(evals > atol) else 0.0
            if min_pos > 0 and max_abs > 0:
                cond_est = float(max_abs / min_pos)
            else:
                cond_est = float('inf')
            shift = float(max(0.0, -min_eig + atol))
            report = {
                "shape": (n, m),
                "is_symmetric": sym,
                "is_psd": psd,
                "min_eigenvalue": min_eig,
                "negative_eigen_count": neg_count,
                "trace": trace,
                "frobenius_norm": frob,
                "cond_estimate": cond_est,
                "recommended_shift": shift
            }
            if report_path:
                try:
                    lines = []
                    lines.append(f"# {title}\n")
                    lines.append(f"- shape: {n}x{m}")
                    lines.append(f"- symmetric: {'yes' if sym else 'no'}")
                    lines.append(f"- PSD: {'yes' if psd else 'no'}")
                    lines.append(f"- min eigenvalue: {min_eig:.6e}")
                    lines.append(f"- negative eigen count(<-atol): {neg_count}")
                    lines.append(f"- trace: {trace:.6e}")
                    lines.append(f"- frobenius norm: {frob:.6e}")
                    lines.append(f"- cond estimate: {cond_est if np.isfinite(cond_est) else 'inf'}")
                    lines.append(f"- recommended shift δ: {shift:.6e}")
                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                    self._qprint(f"[💾] WLTest kernel report saved: {report_path}")
                except Exception as e:
                    self._qprint(f"[⚠️] Failed to write kernel report: {e}")
            return report
        except Exception as e:
            self._qprint(f"[❌] Kernel matrix validation failed: {e}")
            return {"error": str(e)}

    def validate_distance_matrix(self,
                                 distance_matrix: np.ndarray,
                                 report_path: Optional[str] = None,
                                 title: str = "RGWL kernel-induced distance matrix check",
                                 atol: float = 1e-8) -> Dict[str, Any]:
        
        try:
            D = np.asarray(distance_matrix, dtype=float)
            n, m = D.shape
            sym = bool(np.allclose(D, D.T, atol=atol))
            nonneg = bool(np.all(D >= -atol))
            diag_zero = bool(np.allclose(np.diag(D), 0.0, atol=1e-6))
            
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=min(n, 50), replace=False)
            violated = 0
            checked = 0
            for i in idx:
                for j in idx:
                    if j <= i:
                        continue
                    for k in idx:
                        if k == i or k == j:
                            continue
                        checked += 1
                        if D[i, j] - (D[i, k] + D[k, j]) > 1e-6:
                            violated += 1
            tri_ok = (violated == 0)
            report = {
                "shape": (n, m),
                "is_symmetric": sym,
                "is_nonnegative": nonneg,
                "diag_zero": diag_zero,
                "triangle_checked": checked,
                "triangle_violations": violated,
                "triangle_ok": tri_ok
            }
            if report_path:
                try:
                    lines = []
                    lines.append(f"# {title}\n")
                    lines.append(f"- shape: {n}x{m}")
                    lines.append(f"- symmetric: {'yes' if sym else 'no'}")
                    lines.append(f"- nonnegative: {'yes' if nonneg else 'no'}")
                    lines.append(f"- diag zero: {'yes' if diag_zero else 'no'}")
                    lines.append(f"- triangle inequality samples: {checked}, violations: {violated}")
                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                    self._qprint(f"[💾] Distance matrix report saved: {report_path}")
                except Exception as e:
                    self._qprint(f"[⚠️] Failed to write distance report: {e}")
            return report
        except Exception as e:
            self._qprint(f"[❌] Distance matrix validation failed: {e}")
            return {"error": str(e)}
    
    def _extract_answer_subgraph(self, graph_data: Dict[str, Any], graph_id: str) -> Dict[str, Any]:
        self.computation_stats["subgraph_extractions"] += 1
        cache_key = f"subgraph_{graph_id}_{hash(str(graph_data.get('metadata', {})))}"
        if cache_key in self.subgraph_cache:
            self.computation_stats["cache_hits"] += 1
            return self.subgraph_cache[cache_key]
        
        metadata = graph_data.get("metadata", {})
        answer_paths = metadata.get("answer_paths", [])
        nodes_data = graph_data.get("nodes", {})
        edges = graph_data.get("edges", [])
        
        
        if isinstance(nodes_data, list):
            nodes_dict = {}
            for node in nodes_data:
                if isinstance(node, dict) and "id" in node:
                    nodes_dict[node["id"]] = node
            nodes_data = nodes_dict
        if not answer_paths:
            self._qprint(f"[⚠️] {graph_id}: no answer paths, returning empty subgraph")
            return {"nodes": {}, "edges": [], "metadata": {"lambda_ratio": 0.0}}
        path_nodes = set()
        for path in answer_paths:
            path_nodes.update(path)
        
        subgraph_nodes = {nid: nodes_data[nid] for nid in path_nodes if nid in nodes_data}
        subgraph_edges = [(u, v) for u, v in edges if u in path_nodes and v in path_nodes]
        
        total_nodes = len(nodes_data)
        subgraph_lambda = len(path_nodes) / total_nodes if total_nodes > 0 else 0.0
        
        subgraph = {
            "nodes": subgraph_nodes,
            "edges": subgraph_edges,
            "metadata": {
                "lambda_ratio": subgraph_lambda,
                "path_nodes": list(path_nodes),
                "answer_paths": answer_paths,
                "original_graph_id": graph_id
            }
        }
        
        self.subgraph_cache[cache_key] = subgraph
        
        self._qprint(f"[📊] {graph_id} answer subgraph: {len(subgraph_nodes)} nodes, {len(subgraph_edges)} edges, λ={subgraph_lambda:.4f}")
        
        return subgraph
    
    def _label_nodes_knn(self, graph_data: Dict[str, Any], graph_id: str) -> Dict[str, Any]:
        cache_key = f"labels_{graph_id}_{hash(str(graph_data.get('nodes', {})))}"
        if cache_key in self.label_cache:
            self.computation_stats["cache_hits"] += 1
            return self.label_cache[cache_key]
        
        nodes_data = graph_data.get("nodes", {})
        
        if not nodes_data:
            return graph_data
        
        if isinstance(nodes_data, list):
            nodes_dict = {}
            for node in nodes_data:
                if isinstance(node, dict) and "id" in node:
                    nodes_dict[node["id"]] = node
            nodes_data = nodes_dict
        node_contents = []
        node_ids = list(nodes_data.keys())
        
        for nid in node_ids:
            node = nodes_data[nid]
            if isinstance(node, dict):
                content = node.get("content", "")
            else:
                content = getattr(node, "content", "")
            node_contents.append(str(content))
        
        if not any(content.strip() for content in node_contents):
            labeled_graph = graph_data.copy()
            for nid in node_ids:
                if isinstance(labeled_graph["nodes"][nid], dict):
                    labeled_graph["nodes"][nid]["cluster_label"] = 0
                else:
                    labeled_graph["nodes"][nid].cluster_label = 0
            
            self.label_cache[cache_key] = labeled_graph
            return labeled_graph
        
        try:
            if len(node_contents) == 1:
                cluster_labels = [0]
            else:
                tfidf_features = self.tfidf_vectorizer.fit_transform(node_contents)
                X = tfidf_features.toarray()
                n = X.shape[0]
                proto_k = max(2, min(8, n // 2))
                try:
                    km = KMeans(n_clusters=min(proto_k, n), random_state=42, n_init=10)
                    proto_ids = km.fit_predict(X)
                    from sklearn.metrics import pairwise_distances_argmin_min
                    centers = km.cluster_centers_
                    closest, _ = pairwise_distances_argmin_min(centers, X)
                    prototypes = list(set(int(i) for i in closest))
                except Exception:
                    prototypes = list(range(min(3, n)))

                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=min(5, n), metric='cosine')
                nbrs.fit(X)
                cluster_labels = [0] * n
    
                for lab, idx in enumerate(prototypes):
                    cluster_labels[idx] = lab
             
                for i in range(n):
                    if i in prototypes:
                        continue
                    distances, indices = nbrs.kneighbors([X[i]], return_distance=True)
                    neighbor_idxs = list(indices[0])
                    proto_hits = [(j, cluster_labels[j]) for j in neighbor_idxs if j in prototypes]
                    if proto_hits:
                        nearest_proto = min(proto_hits, key=lambda t: distances[0][neighbor_idxs.index(t[0])])
                        cluster_labels[i] = int(nearest_proto[1])
                    else:
                        labs = [cluster_labels[j] for j in neighbor_idxs]
                        cluster_labels[i] = int(Counter(labs).most_common(1)[0][0])
        except Exception as e:
            self._qprint(f"[❌] KNN labeling failed: {e}, using default label")
            cluster_labels = [0] * len(node_contents)
        
        labeled_graph = graph_data.copy()
        labeled_graph["nodes"] = {}
        
        for i, nid in enumerate(node_ids):
            node_copy = nodes_data[nid].copy() if isinstance(nodes_data[nid], dict) else nodes_data[nid].__dict__.copy()
            if isinstance(node_copy, dict):
                node_copy["cluster_label"] = int(cluster_labels[i])
            else:
                if hasattr(node_copy, '__dict__'):
                    node_copy = node_copy.__dict__.copy()
                    node_copy["cluster_label"] = int(cluster_labels[i])
                else:
                    node_copy = {"cluster_label": int(cluster_labels[i]), "original": node_copy}
            
            labeled_graph["nodes"][nid] = node_copy
        
        self.label_cache[cache_key] = labeled_graph
        
        unique_labels = len(set(cluster_labels))
        self._qprint(f"[🏷️] {graph_id} node labeling: {len(node_ids)} nodes, {unique_labels} clusters")
        
        return labeled_graph
    
    def _directed_label_propagation(self, 
                                   labeled_graph: Dict[str, Any], 
                                   round_num: int,
                                   prop_id: str) -> Dict[str, Any]:
        self.computation_stats["label_propagations"] += 1
        
        nodes_data = labeled_graph.get("nodes", {})
        edges = labeled_graph.get("edges", [])
        
        if not nodes_data:
            return labeled_graph
        
        G = nx.DiGraph()
        for nid, node in nodes_data.items():
            label = node.get("cluster_label", 0) if isinstance(node, dict) else getattr(node, "cluster_label", 0)
            G.add_node(nid, label=label, round_0_label=label)
        
        for u, v in edges:
            if u in G.nodes and v in G.nodes:
                G.add_edge(u, v)
        
        for round_i in range(1, round_num + 1):
            new_labels = {}
            
            for node in G.nodes():
                current_label = G.nodes[node][f"round_{round_i-1}_label"]
                
                predecessor_labels = []
                for pred in G.predecessors(node):
                    pred_label = G.nodes[pred][f"round_{round_i-1}_label"]
                    predecessor_labels.append(pred_label)
                
                successor_labels = []
                for succ in G.successors(node):
                    succ_label = G.nodes[succ][f"round_{round_i-1}_label"]
                    successor_labels.append(succ_label)

                all_labels = [current_label] + predecessor_labels + successor_labels
                
                if all_labels:
                    label_counts = Counter(all_labels)
                    most_common = label_counts.most_common(1)[0]
                    
                    if list(label_counts.values()).count(most_common[1]) > 1:
                        label_string = "_".join(map(str, sorted(all_labels)))
                        new_label = hash(label_string) % 1000
                    else:
                        new_label = most_common[0]
                else:
                    new_label = current_label
                
                new_labels[node] = new_label
            for node, label in new_labels.items():
                G.nodes[node][f"round_{round_i}_label"] = label
        
        propagated_graph = labeled_graph.copy()
        propagated_graph["nodes"] = {}
        
        for nid, node_data in nodes_data.items():
            if nid in G.nodes:
                node_copy = node_data.copy() if isinstance(node_data, dict) else node_data.__dict__.copy()
                if isinstance(node_copy, dict):
                    node_copy[f"propagated_label_r{round_num}"] = G.nodes[nid][f"round_{round_num}_label"]
                    for r in range(round_num + 1):
                        node_copy[f"round_{r}_label"] = G.nodes[nid][f"round_{r}_label"]
                else:
                    if not isinstance(node_copy, dict):
                        node_copy = {"original": node_copy}
                    node_copy[f"propagated_label_r{round_num}"] = G.nodes[nid][f"round_{round_num}_label"]
                
                propagated_graph["nodes"][nid] = node_copy
        
        self._qprint(f"[🔄] {prop_id} label propagation round {round_num} completed: {len(G.nodes)} nodes")
        
        return propagated_graph
    
    def _compute_label_histogram(self, propagated_graph: Dict[str, Any], hist_id: str) -> Dict[int, int]:
        nodes_data = propagated_graph.get("nodes", {})
        labels = []
        for nid, node in nodes_data.items():
            propagated_keys = [k for k in node.keys() if k.startswith("propagated_label_r")]
            if propagated_keys:
                latest_key = max(propagated_keys, key=lambda x: int(x.split("_r")[1]))
                label = node[latest_key]
            else:
                label = node.get("cluster_label", 0)
            
            labels.append(label)
        
        histogram = Counter(labels)
        
        self._qprint(f"[📊] {hist_id} histogram: {len(histogram)} distinct labels, total nodes {len(labels)}")
        
        return dict(histogram)
    
    def _compute_inner_product(self, hist1: Dict[int, int], hist2: Dict[int, int]) -> float:
        if not hist1 or not hist2:
            return 0.0
        all_labels = set(hist1.keys()) | set(hist2.keys())
        
        inner_product = 0.0
        for label in all_labels:
            count1 = hist1.get(label, 0)
            count2 = hist2.get(label, 0)
            inner_product += count1 * count2
        
        return inner_product
    
    def _analyze_graph_properties(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        nodes_data = graph_data.get("nodes", {})
        edges = graph_data.get("edges", [])
        metadata = graph_data.get("metadata", {})
        
        if isinstance(nodes_data, list):
            nodes_dict = {}
            for node in nodes_data:
                if isinstance(node, dict) and "id" in node:
                    nodes_dict[node["id"]] = node
            nodes_data = nodes_dict
        
        G = nx.DiGraph()
        G.add_nodes_from(nodes_data.keys())
        G.add_edges_from(edges)
        
        try:
            properties = {
                "node_count": len(nodes_data),
                "edge_count": len(edges),
                "density": nx.density(G),
                "is_connected": nx.is_weakly_connected(G),
                "strongly_connected_components": nx.number_strongly_connected_components(G),
                "weakly_connected_components": nx.number_weakly_connected_components(G),
                "average_in_degree": sum(dict(G.in_degree()).values()) / len(G.nodes()) if G.nodes() else 0,
                "average_out_degree": sum(dict(G.out_degree()).values()) / len(G.nodes()) if G.nodes() else 0,
                "lambda_ratio": metadata.get("lambda_ratio", 0.0),
                "answer_nodes": len(metadata.get("answer_nodes", [])),
                "terminated_nodes": len(metadata.get("terminated_nodes", [])),
                "total_rounds": len(metadata.get("round_nodes", {})) - 1 if metadata.get("round_nodes") else 0
            }
            if metadata.get("answer_paths"):
                path_lengths = [len(path) for path in metadata["answer_paths"]]
                properties["average_path_length"] = np.mean(path_lengths)
                properties["max_path_length"] = max(path_lengths)
                properties["min_path_length"] = min(path_lengths)
            else:
                properties["average_path_length"] = 0
                properties["max_path_length"] = 0
                properties["min_path_length"] = 0
                
        except Exception as e:
            self._qprint(f"[❌] Graph property analysis failed: {e}")
            properties = {
                "node_count": len(nodes_data),
                "edge_count": len(edges),
                "lambda_ratio": metadata.get("lambda_ratio", 0.0)
            }
        
        return properties
    
    def visualize_kernel_computation(self, 
                                   result: Dict[str, Any], 
                                   output_path: str = None,
                                   show_intermediate: bool = True):
        
        if not output_path:
            output_path = f"rgwl_kernel_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'RGWL kernel visualization\nFinal kernel: {result["kernel_value"]:.6f}', fontsize=16)
        
        intermediate = result.get("intermediate_results", {})
        round_contributions = intermediate.get("round_contributions", [])
        
        if round_contributions:
            
            rounds = [r["round"] for r in round_contributions]
            answer_contribs = [r["answer_contribution"] for r in round_contributions]
            full_contribs = [r["full_contribution"] for r in round_contributions]
            cumulative = [r["cumulative_kernel"] for r in round_contributions]
            
            axes[0, 0].bar(rounds, answer_contribs, alpha=0.7, label='answer term', color='skyblue')
            axes[0, 0].bar(rounds, full_contribs, bottom=answer_contribs, alpha=0.7, label='full term', color='lightcoral')
            axes[0, 0].set_xlabel('round')
            axes[0, 0].set_ylabel('contribution')
            axes[0, 0].set_title('per-round contribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(rounds, cumulative, marker='o', linewidth=2, markersize=8, color='green')
            axes[0, 1].set_xlabel('round')
            axes[0, 1].set_ylabel('cumulative kernel')
            axes[0, 1].set_title('cumulative kernel over rounds')
            axes[0, 1].grid(True, alpha=0.3)
            
            
            lambda1 = intermediate.get("lambda1", 0)
            lambda2 = intermediate.get("lambda2", 0)
            lambda_product = lambda1 * lambda2
            
            lambda_data = ['λ1', 'λ2', 'λ1×λ2']
            lambda_values = [lambda1, lambda2, lambda_product]
            axes[0, 2].bar(lambda_data, lambda_values, color=['orange', 'purple', 'red'], alpha=0.7)
            axes[0, 2].set_ylabel('lambda')
            axes[0, 2].set_title('lambda parameters')
            axes[0, 2].grid(True, alpha=0.3)
            
        g1_props = result["graph_properties"]["G1"]
        g2_props = result["graph_properties"]["G2"]
        
        properties = ['node_count', 'edge_count', 'lambda_ratio', 'average_path_length']
        g1_values = [g1_props.get(p, 0) for p in properties]
        g2_values = [g2_props.get(p, 0) for p in properties]
        
        x = np.arange(len(properties))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, g1_values, width, label='Graph1', alpha=0.7, color='lightblue')
        axes[1, 0].bar(x + width/2, g2_values, width, label='Graph2', alpha=0.7, color='lightgreen')
        axes[1, 0].set_xlabel('graph properties')
        axes[1, 0].set_ylabel('value')
        axes[1, 0].set_title('graph property comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(properties, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
        stats = result["computation_stats"]
        stat_names = list(stats.keys())
        stat_values = list(stats.values())
        
        axes[1, 1].pie(stat_values, labels=stat_names, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('computation stats distribution')
        
        answer_total = result["answer_term_sum"]
        full_total = result["full_term_sum"]
        
        components = ['answer term sum', 'full term sum']
        values = [answer_total, full_total]
        colors = ['skyblue', 'lightcoral']
        
        axes[1, 2].pie(values, labels=components, autopct='%1.3f', colors=colors, startangle=90)
        axes[1, 2].set_title('kernel composition')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self._qprint(f"[💾] RGWL kernel visualization saved: {output_path}")
    
    def batch_compute_similarity_matrix(self, graphs: List[Dict[str, Any]], h: int = 3) -> np.ndarray:
        n = len(graphs)
        similarity_matrix = np.zeros((n, n))
        
        self._qprint(f"[🧮] Computing {n}x{n} similarity matrix")
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    result = self.compute_kernel(graphs[i], graphs[j], h)
                    similarity_matrix[i, j] = result["kernel_value"]
                    similarity_matrix[j, i] = similarity_matrix[i, j]
                
                self._qprint(f"[📊] ({i+1},{j+1}): {similarity_matrix[i, j]:.6f}")
        
        return similarity_matrix
    
    def batch_compute_kernel_matrix(self, graphs: List[Dict[str, Any]], h: int = 3) -> np.ndarray:
        n = len(graphs)
        kernel_matrix = np.zeros((n, n), dtype=float)
        self._qprint(f"[🧮] Computing {n}x{n} kernel matrix")
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    result = self.compute_kernel(graphs[i], graphs[i], h)
                    kernel_matrix[i, j] = result["kernel_value"]
                else:
                    result = self.compute_kernel(graphs[i], graphs[j], h)
                    kernel_matrix[i, j] = result["kernel_value"]
                    kernel_matrix[j, i] = kernel_matrix[i, j]
                self._qprint(f"[📊] K({i+1},{j+1}) = {kernel_matrix[i, j]:.6f}")
        return kernel_matrix
    
    def kernel_to_distance_matrix(self, kernel_matrix: np.ndarray) -> np.ndarray:
        diag = np.diag(kernel_matrix)
        d2 = diag[:, None] + diag[None, :] - 2.0 * kernel_matrix
        d2 = np.maximum(d2, 0.0)
        return np.sqrt(d2)
    
    def batch_compute_distance_matrix(self, graphs: List[Dict[str, Any]], h: int = 3) -> np.ndarray:
        K = self.batch_compute_kernel_matrix(graphs, h)
        return self.kernel_to_distance_matrix(K)
    
    def save_computation_cache(self, filepath: str):
        cache_data = {
            "node_feature_cache": self.node_feature_cache,
            "label_cache": self.label_cache,
            "subgraph_cache": self.subgraph_cache,
            "computation_stats": self.computation_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f)
        
        self._qprint(f"[💾] Computation cache saved: {filepath}")
    
    def load_computation_cache(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.node_feature_cache = cache_data.get("node_feature_cache", {})
            self.label_cache = cache_data.get("label_cache", {})
            self.subgraph_cache = cache_data.get("subgraph_cache", {})
            self.computation_stats = cache_data.get("computation_stats", self.computation_stats)
            
            self._qprint(f"[📂] Computation cache loaded: {filepath}")
            
        except Exception as e:
            self._qprint(f"[❌] Failed to load cache: {e}")

 
def create_enhanced_rgwl_kernel(config: Dict[str, Any] = None) -> EnhancedRGWLKernel:
    
    if config is None:
        config = {
            "max_features": 200,
            "n_clusters": 10
        }
    
    return EnhancedRGWLKernel(config)

if __name__ == "__main__":
    
    kernel = create_enhanced_rgwl_kernel()
    
    
    test_graph1 = {
        "nodes": {
            "root": {"content": "problem: 24 points", "cluster_label": 0},
            "node1": {"content": "try addition", "cluster_label": 1},
            "node2": {"content": "3+3+6+12=24", "cluster_label": 2}
        },
        "edges": [("root", "node1"), ("node1", "node2")],
        "metadata": {
            "lambda_ratio": 0.8,
            "answer_paths": [["root", "node1", "node2"]],
            "answer_nodes": ["node2"]
        }
    }
    test_graph2 = {
        "nodes": {
            "root": {"content": "problem: 24 points", "cluster_label": 0},
            "node1": {"content": "use multiplication", "cluster_label": 1},
            "node2": {"content": "6×4=24", "cluster_label": 2}
        },
        "edges": [("root", "node1"), ("node1", "node2")],
        "metadata": {
            "lambda_ratio": 0.9,
            "answer_paths": [["root", "node1", "node2"]],
            "answer_nodes": ["node2"]
        }
    }
    
    
    result = kernel.compute_kernel(test_graph1, test_graph2, h=3, save_intermediate=True)
    
    if os.environ.get('BENCHMARK_QUIET') != '1':
        print("\n=== Test Result ===")
        print(f"Kernel value: {result['kernel_value']}")
        print(f"Answer term sum: {result['answer_term_sum']}")
        print(f"Full term sum: {result['full_term_sum']}")
    
    
    kernel.visualize_kernel_computation(result)