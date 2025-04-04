def create_adjacency_list(edges_file_path: str) -> dict:
    """
    Reads an undirected, unweighted graph from a file and returns its adjacency list representation.

    Args:
        edges_file_path (str): Path to the file containing edges.

    Returns:
        dict: A dictionary where keys are vertices and values are lists of connected vertices.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid data.
    """
    adjacency_list = {}
    try:
        with open(edges_file_path, 'r') as file:
            for line in file:
                try:
                    u, v = map(int, line.strip().split())
                    if u not in adjacency_list:
                        adjacency_list[u] = []
                    if v not in adjacency_list:
                        adjacency_list[v] = []
                    adjacency_list[u].append(v)
                    if u != v:
                        adjacency_list[v].append(u)
                except ValueError:
                    raise ValueError(f"Invalid line in file: {line.strip()}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {edges_file_path}")
    return adjacency_list


def create_adjacency_matrix(edges_file_path: str) -> list[list[int]]:
    """
    Reads an undirected, unweighted graph from a file and returns its adjacency matrix representation.

    Args:
        edges_file_path (str): Path to the file containing edges.

    Returns:
        list: A 2D list representing the adjacency matrix of the graph.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid data.
    """
    edges = []
    max_vertex = -1
    try:
        with open(edges_file_path, 'r') as file:
            for line in file:
                try:
                    u, v = map(int, line.strip().split())
                    edges.append((u, v))
                    max_vertex = max(max_vertex, u, v)
                except ValueError:
                    raise ValueError(f"Invalid line in file: {line.strip()}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {edges_file_path}")

    # Initialize a (max_vertex + 1) x (max_vertex + 1) matrix with zeros
    adjacency_matrix = [[0] * (max_vertex + 1) for _ in range(max_vertex + 1)]

    for u, v in edges:
        adjacency_matrix[u][v] = 1
        adjacency_matrix[v][u] = 1

    return adjacency_matrix