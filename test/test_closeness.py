"""
Test cases for the closeness_centrality function.
Run with `python -m unittest -v test/test_closeness.py` from root directory.
"""
from unittest import TestCase, main
from closeness import closeness_centrality
import networkx as nx

PLACES = 5  # Number of decimal places for comparison


class TestClosenessCentrality(TestCase):
    def test_line_graph(self):
        """
        Test Case 1: Line Graph
        A -- B -- C -- D
        """
        graph = {
            'A': ['B'],
            'B': ['A', 'C'],
            'C': ['B', 'D'],
            'D': ['C']
        }
        nx_graph = nx.Graph(graph)
        expected = nx.closeness_centrality(nx_graph)
        result = closeness_centrality(graph)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_star_graph(self):
        """
        Test Case 2: Star Graph
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
        expected = nx.closeness_centrality(nx_graph)
        result = closeness_centrality(graph)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_cycle_graph(self):
        """
        Test Case 3: Cycle Graph
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
        expected = nx.closeness_centrality(nx_graph)
        result = closeness_centrality(graph)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_grid_graph(self):
        """
        Test Case 4: Grid Graph
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
        expected = nx.closeness_centrality(nx_graph)
        result = closeness_centrality(graph)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)
    
if __name__ == "__main__":
    main()