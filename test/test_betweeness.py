"""
Test cases for the betweenness_centrality function.
Run with `python -m unittest -v test/test_betweeness.py` from root directory.
"""
from unittest import TestCase, main
from betweenness_centrality import betweenness_centrality
import networkx as nx

PLACES = 5  # Number of decimal places for comparison


class TestBetweennessCentrality(TestCase):
    def test_line_graph_normalized(self):
        """
        Test Case: Line Graph
        A -- B -- C -- D
        """
        graph = {
            'A': ['B'],
            'B': ['A', 'C'],
            'C': ['B', 'D'],
            'D': ['C']
        }
        nx_graph = nx.Graph(graph)
        expected = nx.betweenness_centrality(nx_graph, normalized=True)
        result = betweenness_centrality(graph, normalized=True, directed=False)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_line_graph_unnormalized(self):
        """
        Test Case: Line Graph (unnormalized)
        A -- B -- C -- D
        """
        graph = {
            'A': ['B'],
            'B': ['A', 'C'],
            'C': ['B', 'D'],
            'D': ['C']
        }
        nx_graph = nx.Graph(graph)
        expected = nx.betweenness_centrality(nx_graph, normalized=False)
        result = betweenness_centrality(graph, normalized=False, directed=False)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_star_graph_normalized(self):
        """
        Test Case: Star Graph
        B, C, D, E all connect to central node A
        """
        graph = {
            'A': ['B', 'C', 'D', 'E'],
            'B': ['A'],
            'C': ['A'],
            'D': ['A'],
            'E': ['A']
        }
        nx_graph = nx.Graph(graph)
        expected = nx.betweenness_centrality(nx_graph, normalized=True)
        result = betweenness_centrality(graph, normalized=True, directed=False)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_star_graph_unnormalized(self):
        """
        Test Case: Star Graph (unnormalized)
        B, C, D, E all connect to central node A
        """
        graph = {
            'A': ['B', 'C', 'D', 'E'],
            'B': ['A'],
            'C': ['A'],
            'D': ['A'],
            'E': ['A']
        }
        nx_graph = nx.Graph(graph)
        expected = nx.betweenness_centrality(nx_graph, normalized=False)
        result = betweenness_centrality(graph, normalized=False, directed=False)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_cycle_graph_normalized(self):
        """
        Test Case: Cycle Graph
        A -- B
        |    |
        D -- C
        """
        graph = {
            'A': ['B', 'D'],
            'B': ['A', 'C'],
            'C': ['B', 'D'],
            'D': ['A', 'C']
        }
        nx_graph = nx.Graph(graph)
        expected = nx.betweenness_centrality(nx_graph, normalized=True)
        result = betweenness_centrality(graph, normalized=True, directed=False)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_cycle_graph_unnormalized(self):
        """
        Test Case: Cycle Graph (unnormalized)
        A -- B
        |    |
        D -- C
        """
        graph = {
            'A': ['B', 'D'],
            'B': ['A', 'C'],
            'C': ['B', 'D'],
            'D': ['A', 'C']
        }
        nx_graph = nx.Graph(graph)
        expected = nx.betweenness_centrality(nx_graph, normalized=False)
        result = betweenness_centrality(graph, normalized=False, directed=False)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_grid_graph_normalized(self):
        """
        Test Case: Grid Graph
        A -- B -- C
        |    |    |
        D -- E -- F
        """
        graph = {
            'A': ['B', 'D'],
            'B': ['A', 'C', 'E'],
            'C': ['B', 'F'],
            'D': ['A', 'E'],
            'E': ['B', 'D', 'F'],
            'F': ['C', 'E']
        }
        nx_graph = nx.Graph(graph)
        expected = nx.betweenness_centrality(nx_graph, normalized=True)
        result = betweenness_centrality(graph, normalized=True, directed=False)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)
    
    def test_grid_graph_unnormalized(self):
        """
        Test Case: Grid Graph (unnormalized)
        A -- B -- C
        |    |    |
        D -- E -- F
        """
        graph = {
            'A': ['B', 'D'],
            'B': ['A', 'C', 'E'],
            'C': ['B', 'F'],
            'D': ['A', 'E'],
            'E': ['B', 'D', 'F'],
            'F': ['C', 'E']
        }
        nx_graph = nx.Graph(graph)
        expected = nx.betweenness_centrality(nx_graph, normalized=False)
        result = betweenness_centrality(graph, normalized=False)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_directed_graph_normalized(self):
        """
        Test Case: Directed Line Graph
        A -> B -> C -> D
        """
        graph = {
            'A': ['B'],
            'B': ['C'],
            'C': ['D'],
            'D': []
        }
        nx_graph = nx.DiGraph(graph)
        expected = nx.betweenness_centrality(nx_graph, normalized=True)
        result = betweenness_centrality(graph, normalized=True, directed=True)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_directed_graph_unnormalized(self):
        """
        Test Case: Directed Line Graph (unnormalized)
        A -> B -> C -> D
        """
        graph = {
            'A': ['B'],
            'B': ['C'],
            'C': ['D'],
            'D': []
        }
        nx_graph = nx.DiGraph(graph)
        expected = nx.betweenness_centrality(nx_graph, normalized=False)
        result = betweenness_centrality(graph, normalized=False, directed=True)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)


if __name__ == "__main__":
    main()