from __future__ import annotations

from collections import Counter
from typing import Dict

import networkx as nx
import numpy as np


class GraphStatistics:
    """Compute graph metrics used in SpeechGraph."""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def statistics(self) -> Dict[str, float]:
        g = self.graph

        # Handle empty graphs robustly
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()

        if n_nodes == 0:
            return {
                "Nodes": 0,
                "Edges": 0,
                "Parallel Edges": 0,
                "Largest Strongly Connected Component (LSC)": 0,
                "Average Total Degree (ATD)": 0.0,
                "Loops (L1)": 0.0,
                "Loops (L2)": 0.0,
                "Loops (L3)": 0.0,
            }

        # Parallel edges: count how many (u,v) pairs occur more than once
        edge_counts = Counter(g.edges())
        parallel_edges = sum(1 for c in edge_counts.values() if c > 1)

        # Strongly connected components
        scc_sizes = [len(c) for c in nx.strongly_connected_components(g)]
        lsc = max(scc_sizes) if scc_sizes else 0

        degrees = [d for _, d in g.degree()]
        atd = float(np.mean(degrees)) if degrees else 0.0

        # Adjacency matrix (note: MultiDiGraph -> aggregated adjacency)
        A = nx.to_numpy_array(g, dtype=float)
        A2 = A @ A
        A3 = A2 @ A

        l1 = float(np.trace(A))
        l2 = float(np.trace(A2))
        l3 = float(np.trace(A3))

        return {
            "Nodes": int(n_nodes),
            "Edges": int(n_edges),
            "Parallel Edges": int(parallel_edges),
            "Largest Strongly Connected Component (LSC)": int(lsc),
            "Average Total Degree (ATD)": float(atd),
            "Loops (L1)": l1,
            "Loops (L2)": l2,
            "Loops (L3)": l3,
        }