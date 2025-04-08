"""
This file contains unit tests for the eigenvector_centrality function.
Run with `python -m unittest -v test/test_eigenvector.py` from the root directory.
"""

from unittest import TestCase, main
import numpy as np
import networkx as nx
from eigenvector import eigenvector_centrality

PLACES = 6  # Number of decimal places for comparison


class TestEigenvectorCentrality(TestCase):
    def test_simple_2_node_graph(self):
        """
        Test Case 1: Simple 2-node Graph
        A -- B
        Both nodes should have equal centrality.
        """
        graph = np.array([
            [0, 1],
            [1, 0]
        ])
        result = eigenvector_centrality(graph)
        
        # Verify using NetworkX
        nx_graph = nx.from_numpy_array(graph)
        nx_result = nx.eigenvector_centrality(nx_graph)
        
        # Compare dictionaries directly
        self.assertEqual(set(result.keys()), set(nx_result.keys()))
        for node in result:
            self.assertAlmostEqual(result[node], nx_result[node], places=PLACES)

    def test_triangle_graph(self):
        """
        Test Case 2: Triangle Graph
        A -- B
        |    |
        C -- D
        All nodes should have equal centrality.
        """
        graph = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        result = eigenvector_centrality(graph)
        
        # Verify using NetworkX
        nx_graph = nx.from_numpy_array(graph)
        nx_result = nx.eigenvector_centrality(nx_graph)
        
        # Compare dictionaries directly
        self.assertEqual(set(result.keys()), set(nx_result.keys()))
        for node in result:
            self.assertAlmostEqual(result[node], nx_result[node], places=PLACES)


    def test_tree_like_graph(self):
        """
        Test Case 3: Tree-like Graph
        A -- C
        B -- C
        Node C should have higher centrality than nodes A and B.
        """
        graph = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0]
        ])
        result = eigenvector_centrality(graph)
        
        # Verify using NetworkX
        nx_graph = nx.from_numpy_array(graph)
        nx_result = nx.eigenvector_centrality(nx_graph)
        
        # Compare dictionaries directly
        self.assertEqual(set(result.keys()), set(nx_result.keys()))
        for node in result:
            self.assertAlmostEqual(result[node], nx_result[node], places=PLACES)

    def test_star_graph(self):
        """
        Test Case 4: Star Graph
        Center node (0) connects to all others, others only connect to node 0.
        Center node should have the highest centrality.
        """
        graph = np.array([
            [0, 1, 1, 1],  
            [1, 0, 0, 0],  
            [1, 0, 0, 0],  
            [1, 0, 0, 0]   
        ])
        result = eigenvector_centrality(graph)
        
        # Verify using NetworkX
        nx_graph = nx.from_numpy_array(graph)
        nx_result = nx.eigenvector_centrality(nx_graph)
        
        # Compare dictionaries directly
        self.assertEqual(set(result.keys()), set(nx_result.keys()))
        for node in result:
            self.assertAlmostEqual(result[node], nx_result[node], places=PLACES)

    def test_disconnected_graph(self):
        """
        Test Case 5: 
        A: B, C
        B: A
        C: A, D
        D: A, C
        """
        graph = np.array([
            [0, 1, 1, 1],  
            [1, 0, 0, 0],  
            [1, 0, 0, 1],  
            [1, 0, 1, 0]   
        ])
        result = eigenvector_centrality(graph)
        
        # Verify using NetworkX
        nx_graph = nx.from_numpy_array(graph)
        nx_result = nx.eigenvector_centrality(nx_graph)
        
        # Compare dictionaries directly
        self.assertEqual(set(result.keys()), set(nx_result.keys()))
        for node in result:
            self.assertAlmostEqual(result[node], nx_result[node], places=PLACES)

if __name__ == "__main__":
    main()
