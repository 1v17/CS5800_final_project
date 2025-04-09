from collections import deque

def closeness_centrality(graph):
    """
    Compute closeness centrality for all nodes in an unweighted undirected graph.

    Args:
        graph (dict): Adjacency list representation of the graph (e.g., {0: [1, 2], 1: [0], 2: [0]}).

    Returns:
        dict: A dictionary mapping each node to its closeness centrality.
    """
    centrality = {}
    nodes = list(graph.keys())
    n = len(nodes)

    for node in nodes:
        # Compute shortest paths using BFS
        distances = {n: -1 for n in nodes}  # -1 means unreachable
        distances[node] = 0
        queue = deque([node])

        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if distances[neighbor] == -1:  # Not visited yet
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        # Calculate sum of distances (excluding the node itself)
        # Filter out unreachable nodes (-1 values)
        reachable_distances = [d for n, d in distances.items() if d > 0]
        
        # Count number of reachable nodes (excluding the node itself)
        reachable_count = len(reachable_distances)
        
        # Sum of all shortest path distances
        total_distance = sum(reachable_distances)
        
        # Closeness centrality is 0 for isolated nodes
        if reachable_count == 0:
            centrality[node] = 0.0
        else:
            # Standard closeness centrality: (n-1) / sum of distances
            # where n is the number of nodes in the graph
            centrality[node] = (n - 1) / total_distance if total_distance > 0 else 0.0

    return centrality