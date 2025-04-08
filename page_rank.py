from collections import defaultdict

DEFAULT_FACTOR = 0.85
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_CONVERGENCE_THRESHOLD = 1e-06


def page_rank(graph: dict, damping_factor: float=DEFAULT_FACTOR,
              max_iterations: int=DEFAULT_MAX_ITERATIONS, 
              convergence_threshold: float=DEFAULT_CONVERGENCE_THRESHOLD) -> dict:
    """
    Computes the PageRank scores for all nodes in a graph using the power iteration method.

    Args:
        graph (dict): Adjacency list representation of the graph.
                      Keys are nodes, values are lists of neighboring nodes.
        damping_factor (float): Probability of following a link (default: 0.85).
        max_iterations (int): Maximum number of iterations for power iteration (default: 100).
        convergence_threshold (float): Threshold for convergence (default: 1e-06).

    Returns:
        dict: A dictionary mapping each node to its PageRank score.
    """
    if not isinstance(graph, dict):
        raise TypeError("Graph must be a dictionary")
    if not isinstance(damping_factor, (float, int)) or not (0 < damping_factor < 1):
        raise ValueError("Damping factor must be a float between 0 and 1")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("Maximum iterations must be a positive integer")
    if not isinstance(convergence_threshold, (float, int)) or convergence_threshold <= 0:
        raise ValueError("Convergence threshold must be a positive float")
    
    # Step 1: Initialize variables
    num_nodes = len(graph)
    if num_nodes == 0:
        return {}

    # Initialize PageRank scores to 1/N for all nodes
    ranks = {node: 1 / num_nodes for node in graph}

    # Handle dangling nodes (nodes with no outgoing edges)
    dangling_nodes = {node for node,
                      neighbors in graph.items() if len(neighbors) == 0}

    # Step 2: Power iteration
    for iteration in range(max_iterations):
        new_ranks = defaultdict(float)

        # Distribute ranks from each node
        for node in graph:
            # Distribute rank to neighbors
            if node not in dangling_nodes:
                rank_share = ranks[node] / len(graph[node])
                for neighbor in graph[node]:
                    new_ranks[neighbor] += damping_factor * rank_share
            else:
                # Handle dangling nodes by redistributing their rank equally to all nodes
                for other_node in graph:
                    new_ranks[other_node] += damping_factor * \
                        (ranks[node] / num_nodes)

        # Add teleportation (random jump) factor
        for node in graph:
            new_ranks[node] += (1 - damping_factor) / num_nodes

        # Check for convergence
        diff = sum(abs(new_ranks[node] - ranks[node]) for node in graph)
        if diff < convergence_threshold:
            break

        # Update ranks for the next iteration
        ranks = new_ranks

    return dict(ranks)
