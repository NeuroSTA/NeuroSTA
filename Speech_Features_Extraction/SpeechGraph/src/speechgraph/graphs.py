from __future__ import annotations

from re import sub
from typing import List

import networkx as nx


class NaiveGraph:
    """
    Builds a directed MultiDiGraph from a token sequence.
    Nodes: tokens
    Edges: consecutive token transitions
    """

    @staticmethod
    def _clean_and_tokenize(text: str) -> List[str]:
        cleaned = sub(r"[^\w ]+", " ", text.lower().strip())
        tokens = cleaned.split()
        return tokens

    def text_to_graph(self, text: str) -> nx.MultiDiGraph:
        tokens = self._clean_and_tokenize(text)
        g = nx.MultiDiGraph()
        if len(tokens) < 2:
            # If <2 tokens: graph exists but has no edges.
            g.add_nodes_from(tokens)
            return g
        g.add_edges_from(zip(tokens[:-1], tokens[1:]))
        return g