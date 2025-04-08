"""
This file contains unit tests for the PageRank algorithm.
Run with `python -m unittest -v test/test_page_rank.py` from the root directory.
"""
from unittest import TestCase, main
from page_rank import page_rank_centrality
import networkx as nx

PLACES = 5
DEFAULT_FACTOR = 0.85


class TestPageRank(TestCase):

    def test_empty_graph(self):
        """Test PageRank on an empty graph."""
        graph = {}
        result = page_rank_centrality(graph)
        self.assertEqual(result, {})

    def test_single_node_graph(self):
        """Test PageRank on a graph with a single node."""
        graph = {'A': []}
        result = page_rank_centrality(graph)
        expected = {'A': 1.0}
        self.assertEqual(result, expected)

    def test_two_node_graph(self):
        """Test PageRank on a graph with two nodes and one edge."""
        graph = {'A': ['B'], 'B': []}
        result = page_rank_centrality(graph)
        nx_graph = nx.DiGraph(graph)
        expected = nx.pagerank(nx_graph, alpha=DEFAULT_FACTOR)
        self.assertAlmostEqual(result['A'], expected['A'], places=PLACES)
        self.assertAlmostEqual(result['B'], expected['B'], places=PLACES)

    def test_cyclic_graph(self):
        """Test PageRank on a cyclic graph."""
        graph = {'A': ['B'], 'B': ['C'], 'C': ['A']}
        result = page_rank_centrality(graph)
        G = nx.DiGraph(graph)
        expected = nx.pagerank(G, alpha=DEFAULT_FACTOR)
        for node in graph:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_dangling_node(self):
        """Test PageRank on a graph with a dangling node."""
        graph = {'A': ['B'], 'B': ['C'], 'C': []}
        result = page_rank_centrality(graph)
        G = nx.DiGraph(graph)
        expected = nx.pagerank(G, alpha=DEFAULT_FACTOR)
        for node in graph:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_disconnected_graph(self):
        """Test PageRank on a disconnected graph."""
        graph = {'A': ['B'], 'B': [], 'C': ['D'], 'D': []}
        result = page_rank_centrality(graph)
        G = nx.DiGraph(graph)
        expected = nx.pagerank(G, alpha=DEFAULT_FACTOR)
        for node in graph:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_undirected_graph(self):
        """Test PageRank on an undirected graph."""
        graph = {'A': ['B'], 'B': ['A', 'C'], 'C': ['B']}
        result = page_rank_centrality(graph)
        G = nx.Graph(graph)  # Convert to undirected graph
        expected = nx.pagerank(G, alpha=DEFAULT_FACTOR)
        for node in graph:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_custom_damping_factor(self):
        """Test PageRank with a custom damping factor."""
        graph = {'A': ['B'], 'B': ['C'], 'C': ['A']}
        damping_factor = 0.9
        result = page_rank_centrality(graph, damping_factor=damping_factor)
        G = nx.DiGraph(graph)
        expected = nx.pagerank(G, alpha=damping_factor)
        for node in graph:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_custom_convergence_threshold(self):
        """Test PageRank with a custom convergence threshold."""
        graph = {'A': ['B'], 'B': ['C'], 'C': ['A']}
        convergence_threshold = 1e-04
        result = page_rank_centrality(graph, convergence_threshold=convergence_threshold)
        G = nx.DiGraph(graph)
        expected = nx.pagerank(G, alpha=DEFAULT_FACTOR)
        for node in graph:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)


if __name__ == '__main__':
    main()
