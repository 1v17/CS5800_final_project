from utils import create_adjacency_list, get_top_centrality
from betweenness_centrality import betweenness_centrality

PATH = "test/test_files/"


def main():
    # Calculate betweenness centrality for a graph
    graph = create_adjacency_list("facebook_combined.txt")
    centrality = betweenness_centrality(graph, normalized=True, directed=False)
    print("Betweenness Centrality:", centrality)
    print("Top 10 Centrality:", get_top_centrality(centrality, top_n=10))


if __name__ == "__main__":
    main()
