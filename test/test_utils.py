"""
Unit tests for graph conversion functions in utils module.
Run with `python -m unittest test/test_utils.py`.
"""

from unittest import TestCase, main
from utils import create_adjacency_list, create_adjacency_matrix

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
        self.assertEqual(adjacency_matrix, expected)

    def test_create_adjacency_matrix_empty_graph(self):
        adjacency_matrix = create_adjacency_matrix(PATH + "empty_graph.txt")
        expected = []
        self.assertEqual(adjacency_matrix, expected)

    def test_create_adjacency_matrix_single_edge(self):
        adjacency_matrix = create_adjacency_matrix(PATH + "single_edge.txt")
        expected = [
            [0, 1],
            [1, 0]
        ]
        self.assertEqual(adjacency_matrix, expected)

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
        self.assertEqual(adjacency_matrix, expected)

    def test_create_adjacency_matrix_self_loop(self):
        adjacency_matrix = create_adjacency_matrix(PATH + "self_loop.txt")
        expected = [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ]
        self.assertEqual(adjacency_matrix, expected)

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


if __name__ == "__main__":
    main()
