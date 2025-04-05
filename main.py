from utils import create_adjacency_list, get_top_centrality, compare_centrality_with_egos
from betweenness_centrality import betweenness_centrality

DATA_FILE = "facebook_data/facebook_combined.txt"
EGO_VERTICES = set([0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980])


def main():
    # Calculate betweenness centrality for a graph
    graph = create_adjacency_list(DATA_FILE)
    centrality = betweenness_centrality(graph, normalized=True, directed=False)
    top_centrality_nodes = get_top_centrality(centrality, top_n=10)
    print("Top 10 Centrality:", top_centrality_nodes)
    compare_centrality_with_egos(top_centrality_nodes, EGO_VERTICES)


if __name__ == "__main__":
    main()
