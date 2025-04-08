from utils import create_adjacency_list, create_adjacency_matrix, get_top_centrality, compare_centrality_with_egos
from betweenness_centrality import betweenness_centrality
from eigenvector import eigenvector_centrality
from page_rank import page_rank_centrality

DATA_FILE = "facebook_data/facebook_combined.txt"
EGO_VERTICES = set([0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980])


def main():
    try:
        print("Creating Adjacency Matrix and List...")
        adjacency_matrix = create_adjacency_matrix(DATA_FILE)
        adjacency_list = create_adjacency_list(DATA_FILE)

        # Calculate betweenness centrality for a graph
        print("Calculating Betweenness Centrality...")
        b_centrality = betweenness_centrality(adjacency_list, normalized=True, directed=False)
        top_centrality_nodes = get_top_centrality(b_centrality, top_n=10)
        print("Top 10 Centrality:", top_centrality_nodes)
        compare_centrality_with_egos(top_centrality_nodes, EGO_VERTICES)

        # Calculate eigenvector centrality for a graph
        print("\nCalculating Eigenvector Centrality...")
        e_centrality = eigenvector_centrality(adjacency_matrix)
        top_eigenvector_centrality_nodes = get_top_centrality(e_centrality, top_n=10)
        print("Top 10 Eigenvector Centrality:", top_eigenvector_centrality_nodes)
        compare_centrality_with_egos(top_eigenvector_centrality_nodes, EGO_VERTICES)

        # Calculate PageRank centrality for a graph
        print("\nCalculating PageRank Centrality...")
        pagerank_centrality = page_rank_centrality(adjacency_list)
        top_pagerank_centrality_nodes = get_top_centrality(pagerank_centrality, top_n=10)
        print("Top 10 PageRank Centrality:", top_pagerank_centrality_nodes)
        compare_centrality_with_egos(top_pagerank_centrality_nodes, EGO_VERTICES)


    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file path.")
    except ValueError as e:
        print(f"Error: {e}. Please check the file format.")
    except TypeError as e:
        print(f"Error: {e}. Please check the input types.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}.")


if __name__ == "__main__":
    main()
