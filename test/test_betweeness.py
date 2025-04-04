"""
Test cases for the betweenness_centrality function.
Run with `python -m unittest -v test/test_betweeness.py` from root directory.
"""
from unittest import TestCase, main
from betweenness_centrality import betweenness_centrality

PLACES = 6  # Number of decimal places for comparison


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
        expected = {'A': 0.0, 'B': 2/3, 'C': 2/3, 'D': 0.0}
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
        expected = {'A': 0.0, 'B': 2.0, 'C': 2.0, 'D': 0.0}
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
        expected = {'A': 1.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0}
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
        expected = {'A': 6.0, 'B': 0.0, 'C': 0.0, 'D': 0.0, 'E': 0.0}
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
        expected = {'A': 1/6, 'B': 1/6, 'C': 1/6, 'D': 1/6}
        result = betweenness_centrality(graph, normalized=True, directed=False)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)

    def test_cycle_graph_unnormalized(self):
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
        expected = {'A': 0.5, 'B': 0.5, 'C': 0.5, 'D': 0.5}
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
        expected = {'A': 0.0, 'B': 0.25, 'C': 0.0, 'D': 0.25, 'E': 0.5, 'F': 0.0}
        result = betweenness_centrality(graph, normalized=True, directed=False)
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
        expected = {'A': 0.0, 'B': 2/3, 'C': 2/3, 'D': 0.0}
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
        expected = {'A': 0.0, 'B': 2.0, 'C': 2.0, 'D': 0.0}
        result = betweenness_centrality(graph, normalized=False, directed=True)
        for node in expected:
            self.assertAlmostEqual(result[node], expected[node], places=PLACES)


if __name__ == "__main__":
    main()