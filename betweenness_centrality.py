from collections import deque


def bfs_shortest_paths(graph, source):
    """
    Perform BFS to compute shortest paths in an unweighted graph.

    Args:
        graph (dict): Adjacency list representation of the graph.
        source: The source node.

    Returns:
        tuple: (stack, pred, sigma)
            - stack: List of nodes in the order they are processed.
            - pred: Dictionary of predecessors for each node.
            - sigma: Dictionary of the number of shortest paths to each node.
    """
    stack = []
    pred = {v: [] for v in graph}
    sigma = dict.fromkeys(graph, 0.0)
    sigma[source] = 1.0
    dist = dict.fromkeys(graph, -1)
    dist[source] = 0

    queue = deque([source])
    while queue:
        v = queue.popleft()
        stack.append(v)
        for w in graph[v]:
            if dist[w] < 0:  # Found for the first time
                queue.append(w)
                dist[w] = dist[v] + 1
            if dist[w] == dist[v] + 1:  # Shortest path to w via v
                sigma[w] += sigma[v]
                pred[w].append(v)

    return stack, pred, sigma


def accumulate_dependencies(stack, pred, sigma, source):
    """
    Accumulate dependencies to compute betweenness centrality.

    Args:
        stack (list): List of nodes in the order they are processed.
        pred (dict): Dictionary of predecessors for each node.
        sigma (dict): Dictionary of the number of shortest paths to each node.
        source: The source node.

    Returns:
        dict: Dictionary of dependency values for each node.
    """
    delta = dict.fromkeys(pred, 0.0)
    while stack:
        w = stack.pop()
        for v in pred[w]:
            delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
        if w != source:
            yield w, delta[w]


def betweenness_centrality(graph, normalized=True, directed=False):
    """
    Computes the betweenness centrality for all nodes in a graph using Brandes' algorithm.

    Args:
        graph (dict): Adjacency list representation of the graph.
                      Keys are nodes, values are lists of neighboring nodes.
        normalized (bool): Whether to normalize the centrality scores. Default is True.
        directed (bool): Whether the graph is directed. Default is False.
        weight (dict): Optional dictionary of edge weights with (u, v) as keys and weights as values.

    Returns:
        dict: A dictionary mapping each node to its betweenness centrality score.
    """
    betweenness = dict.fromkeys(graph, 0.0)

    for source in graph:

        stack, pred, sigma = bfs_shortest_paths(graph, source)

        for w, delta_w in accumulate_dependencies(stack, pred, sigma, source):
            betweenness[w] += delta_w

    if normalized:
        scale = 1 / ((len(graph) - 1) * (len(graph) - 2)) if len(graph) > 2 else None
        if not directed:
            scale *= 2
        for v in betweenness:
            betweenness[v] *= scale

    return betweenness
