from utils import create_adjacency_list, create_adjacency_matrix, \
    get_top_centrality, compare_centrality_with_egos, \
    plot_social_network_with_centrality, plot_social_network
from betweenness_centrality import betweenness_centrality
from eigenvector import eigenvector_centrality
from page_rank import page_rank_centrality

DATA_FILE = "facebook_data/facebook_combined.txt"
DEFLAULT_NODES = 10
EGO_VERTICES = {0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980}


def main():
    try:
        print("Creating Adjacency Matrix and List...")
        adjacency_matrix = create_adjacency_matrix(DATA_FILE)
        adjacency_list = create_adjacency_list(DATA_FILE)
        plot_social_network(adjacency_list, EGO_VERTICES)
        # TODO: change the main function by using a loop to call other functions

        # Calculate betweenness centrality for a graph
        print("Calculating Betweenness Centrality...")
        b_centrality = betweenness_centrality(adjacency_list, normalized=True, directed=False)
        top_bcentrality_nodes = get_top_centrality(b_centrality, top_n=DEFLAULT_NODES)
        print("Top 10 Centrality:", top_bcentrality_nodes)
        compare_centrality_with_egos(top_bcentrality_nodes, EGO_VERTICES)
        plot_social_network_with_centrality(adjacency_list, b_centrality, "betweenness", top_bcentrality_nodes)

        # Calculate eigenvector centrality for a graph
        # TODO: need to update the eigenvector_centrality function to return a dictionary
        print("\nCalculating Eigenvector Centrality...")
        e_centrality = eigenvector_centrality(adjacency_matrix)
        top_eigenvector_centrality_nodes = get_top_centrality(e_centrality, top_n=DEFLAULT_NODES)
        print("Top 10 Eigenvector Centrality:", top_eigenvector_centrality_nodes)
        compare_centrality_with_egos(top_eigenvector_centrality_nodes, EGO_VERTICES)
        plot_social_network_with_centrality(adjacency_list, e_centrality, "eigenvector", top_eigenvector_centrality_nodes)

        # Calculate PageRank centrality for a graph
        print("\nCalculating PageRank Centrality...")
        pagerank_centrality = page_rank_centrality(adjacency_list)
        top_pagerank_centrality_nodes = get_top_centrality(pagerank_centrality, top_n=DEFLAULT_NODES)
        print("Top 10 PageRank Centrality:", top_pagerank_centrality_nodes)
        compare_centrality_with_egos(top_pagerank_centrality_nodes, EGO_VERTICES)
        plot_social_network_with_centrality(adjacency_list, pagerank_centrality, "page rank", top_pagerank_centrality_nodes)

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
