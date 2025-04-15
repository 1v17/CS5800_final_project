from utils import create_adjacency_list, create_adjacency_matrix, \
    get_top_centrality, compare_centrality_with_egos, \
    plot_social_network_with_centrality, plot_social_network
from betweenness_centrality import betweenness_centrality
from eigenvector import eigenvector_centrality
from page_rank import page_rank_centrality
from closeness import closeness_centrality
import networkx as nx

DATA_FILE = "facebook_data/facebook_combined.txt"
DEFLAULT_NODES = 10
EGO_VERTICES = {0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980}


def main():
    try:
        print("Creating Adjacency Matrix and List...")
        # adjacency_matrix = create_adjacency_matrix(DATA_FILE)
        adjacency_list = create_adjacency_list(DATA_FILE)
        # plot_social_network(adjacency_list, EGO_VERTICES)

        # caluculate average clustering coefficient of the dataset
        nx_graph = nx.Graph(adjacency_list)
        clustering_coefficient = nx.average_clustering(nx_graph)
        print(f"Average Clustering Coefficient: {clustering_coefficient:.4f}")
        random_graph = nx.gnm_random_graph(4039, 88234)
        random_clustering_coefficient = nx.average_clustering(random_graph)
        print(f"Average Clustering Coefficient of Random Graph: {random_clustering_coefficient:.4f}")

        # change this to run different centrality functions
        # centrality_list = ["closeness", "betweenness", "eigenvector", "pagerank"]
        centrality_list = []

        for centrality_measure in centrality_list:
            print(f"\nCalculating {centrality_measure.capitalize()} Centrality...")
            match centrality_measure:
                case "closeness":
                    centrality = closeness_centrality(adjacency_list)
                case "betweenness":
                    centrality = betweenness_centrality(adjacency_list, normalized=True, directed=False)
                case "eigenvector":
                    centrality = eigenvector_centrality(adjacency_matrix)
                case "pagerank":
                    centrality = page_rank_centrality(adjacency_list)
                case _:
                    raise ValueError(f"Unknown centrality measure: {centrality_measure}")

            top_centrality_nodes = get_top_centrality(centrality, top_n=DEFLAULT_NODES)
            print(f"Top 10 {centrality_measure.capitalize()} Centrality:", top_centrality_nodes)
            compare_centrality_with_egos(top_centrality_nodes, EGO_VERTICES)
            plot_social_network_with_centrality(adjacency_list, centrality, centrality_measure, top_centrality_nodes)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file path.")
    except ValueError as e:
        print(f"Error: {e}. Please check the file format.")
    except TypeError as e:
        print(f"Error: {e}. Please check the input types.")
    except KeyError as e:
        print(f"Error: {e}. Please check the graph structure.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}.")


if __name__ == "__main__":
    main()
