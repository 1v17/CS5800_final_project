"""
This file contains unit tests for the eigenvector_centrality function.
Run with `python -m unittest -v test/test_eigenvector.py` from the root directory.
"""

from unittest import TestCase, main
import numpy as np
from eigenvector import eigenvector_centrality

PLACES = 6  # Number of decimal places for comparison


class TestEigenvectorCentrality(TestCase):
    def test_simple_2_node_graph(self):
        """
        Test Case 1: Simple 2-node Graph
        A -- B
        Both nodes should have equal centrality.
        """
        A = np.array([
            [0, 1],
            [1, 0]
        ])
        result = eigenvector_centrality(A)
        self.assertTrue(np.allclose(
            result[0], result[1], rtol=PLACES), "Both nodes should have equal centrality")

    def test_triangle_graph(self):
        """
        Test Case 2: Triangle Graph
        A -- B
        |    |
        C -- D
        All nodes should have equal centrality.
        """
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        result = eigenvector_centrality(A)
        self.assertTrue(np.allclose(
            result, result[0]), "All nodes should have equal centrality")

    def test_tree_like_graph(self):
        """
        Test Case 3: Tree-like Graph
        A -- C
        B -- C
        Node C should have higher centrality than nodes A and B.
        """
        A = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0]
        ])
        result = eigenvector_centrality(A)
        self.assertTrue(result[2] > result[0] and result[2]
                        > result[1], "Node C should have higher centrality")

    def test_star_graph(self):
        """
        Test Case 4: Star Graph
        Center node (0) connects to all others, others only connect to node 0.
        Center node should have the highest centrality.
        """
        A = np.array([
            [0, 1, 1, 1],  # Node 0 connects to everyone
            [1, 0, 0, 0],  # Node 1 only connects to 0
            [1, 0, 0, 0],  # Node 2 only connects to 0
            [1, 0, 0, 0]   # Node 3 only connects to 0
        ])
        result = eigenvector_centrality(A)
        self.assertTrue(result[0] > result[1] and result[0] > result[2] and result[0] > result[3],
                        "Center node should have the highest centrality")


if __name__ == "__main__":
    main()
