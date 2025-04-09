"""
Unit tests for graph conversion functions in utils module.
Run with `python -m unittest -v test/test_utils.py` from root directory.
"""

from unittest import TestCase, main
import numpy as np
from utils import create_adjacency_list, create_adjacency_matrix, get_top_centrality

PATH = "test/test_files/"


class TestGraphConversionFunctions(TestCase):
    def test_create_adjacency_list_small_graph(self):
        adjacency_list = create_adjacency_list(PATH + "small_graph.txt")
        expected = {
            0: [1, 2, 4],
            1: [0, 2],
            2: [0, 1, 3],
            3: [2, 4],
            4: [3, 0]
        }
        self.assertEqual(adjacency_list, expected)

    def test_create_adjacency_list_empty_graph(self):
        adjacency_list = create_adjacency_list(PATH + "empty_graph.txt")
        expected = {}
        self.assertEqual(adjacency_list, expected)

    def test_create_adjacency_list_single_edge(self):
        adjacency_list = create_adjacency_list(PATH + "single_edge.txt")
        expected = {
            0: [1],
            1: [0]
        }
        self.assertEqual(adjacency_list, expected)

    def test_create_adjacency_list_disconnected_graph(self):
        adjacency_list = create_adjacency_list(PATH + "disconnected_graph.txt")
        expected = {
            0: [1],
            1: [0],
            2: [3],
            3: [2],
            4: [5],
            5: [4]
        }
        self.assertEqual(adjacency_list, expected)

    def test_create_adjacency_list_self_loop(self):
        adjacency_list = create_adjacency_list(PATH + "self_loop.txt")
        expected = {
            0: [0],
            1: [2],
            2: [1]
        }
        self.assertEqual(adjacency_list, expected)

    def test_create_adjacency_matrix_small_graph(self):
        adjacency_matrix = create_adjacency_matrix(PATH + "small_graph.txt")
        expected = [
            [0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0]
        ]
        np.testing.assert_array_equal(adjacency_matrix, expected)

    def test_create_adjacency_matrix_empty_graph(self):
        adjacency_matrix = create_adjacency_matrix(PATH + "empty_graph.txt")
        expected = np.zeros((0, 0), dtype=int)
        np.testing.assert_array_equal(adjacency_matrix, expected)

    def test_create_adjacency_matrix_single_edge(self):
        adjacency_matrix = create_adjacency_matrix(PATH + "single_edge.txt")
        expected = [
            [0, 1],
            [1, 0]
        ]
        np.testing.assert_array_equal(adjacency_matrix, expected)

    def test_create_adjacency_matrix_disconnected_graph(self):
        adjacency_matrix = create_adjacency_matrix(
            PATH + "disconnected_graph.txt")
        expected = [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0]
        ]
        np.testing.assert_array_equal(adjacency_matrix, expected)

    def test_create_adjacency_matrix_self_loop(self):
        adjacency_matrix = create_adjacency_matrix(PATH + "self_loop.txt")
        expected = [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ]
        np.testing.assert_array_equal(adjacency_matrix, expected)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            create_adjacency_list("non_existent_file.txt")
        with self.assertRaises(FileNotFoundError):
            create_adjacency_matrix("non_existent_file.txt")

    def test_invalid_line(self):
        with self.assertRaises(ValueError):
            create_adjacency_list(PATH + "invalid_line.txt")
        with self.assertRaises(ValueError):
            create_adjacency_matrix(PATH + "invalid_line.txt")
    
    def test_get_top_centrality_basic(self):
        """
        Test basic functionality with a small centrality dictionary.
        """
        centrality_dict = {'A': 0.5, 'B': 0.8, 'C': 0.3, 'D': 0.9}
        top_n = 2
        expected = [('D', 0.9), ('B', 0.8)]
        result = get_top_centrality(centrality_dict, top_n)
        self.assertEqual(result, expected)

    def test_get_top_centrality_all_nodes(self):
        """
        Test when top_n is equal to the number of nodes in the dictionary.
        """
        centrality_dict = {'A': 0.5, 'B': 0.8, 'C': 0.3}
        top_n = 3
        expected = [('B', 0.8), ('A', 0.5), ('C', 0.3)]
        result = get_top_centrality(centrality_dict, top_n)
        self.assertEqual(result, expected)

    def test_get_top_centrality_more_than_available(self):
        """
        Test when top_n is greater than the number of nodes in the dictionary.
        """
        centrality_dict = {'A': 0.5, 'B': 0.8}
        top_n = 5
        expected = [('B', 0.8), ('A', 0.5)]
        result = get_top_centrality(centrality_dict, top_n)
        self.assertEqual(result, expected)

    def test_get_top_centrality_empty_dict(self):
        """
        Test when the centrality dictionary is empty.
        """
        centrality_dict = {}
        top_n = 3
        expected = []
        result = get_top_centrality(centrality_dict, top_n)
        self.assertEqual(result, expected)

    def test_get_top_centrality_negative_top_n(self):
        """
        Test when top_n is negative.
        """
        centrality_dict = {'A': 0.5, 'B': 0.8}
        top_n = -1
        with self.assertRaises(ValueError):
            get_top_centrality(centrality_dict, top_n)

    def test_get_top_centrality_zero_top_n(self):
        """
        Test when top_n is zero.
        """
        centrality_dict = {'A': 0.5, 'B': 0.8}
        top_n = 0
        with self.assertRaises(ValueError):
            get_top_centrality(centrality_dict, top_n)

    def test_get_top_centrality_non_integer_top_n(self):
        """
        Test when top_n is not an integer.
        """
        centrality_dict = {'A': 0.5, 'B': 0.8}
        top_n = 2.5
        with self.assertRaises(TypeError):
            get_top_centrality(centrality_dict, top_n)

    def test_get_top_centrality_ties(self):
        """
        Test when there are ties in centrality scores.
        """
        centrality_dict = {'A': 0.5, 'B': 0.8, 'C': 0.8, 'D': 0.3}
        top_n = 2
        expected = [('B', 0.8), ('C', 0.8)]  # Order of ties depends on sorting
        result = get_top_centrality(centrality_dict, top_n)
        self.assertEqual(result, expected[:top_n])


if __name__ == "__main__":
    main()
