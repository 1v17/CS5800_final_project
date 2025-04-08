import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

DEFLAULT_NODES = 10
FIGURE_SIZE = 12
LABEL_SIZE = 10
POSITION_SEED = 42
EDGE_COLOR = "gray"
NODE_COLOR = "skyblue"
EGO_NODE_COLOR = "coral"
EDGE_WIDTH = 0.5
NODE_EDGE_COLOR = "black"
LABEL_FONT_COLOR = "darkorange"
EGO_NODE_LABEL_COLOR = "black"
LEGEND_COLOR = "white"
TRANSPARENCY = 0.8
DEFAULT_LAYOUT = "spring"
NODE_SIZE = 100
SAMLL_NODE_SIZE = 10
HIGHLIGHT_NODE_SIZE = 360
COLOR_MAP = cm.viridis
GARPH_PATH = "graphs/"


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


def create_adjacency_matrix(edges_file_path: str) -> np.ndarray:

    """
    Reads an undirected, unweighted graph from a file and returns its adjacency matrix representation.

    Args:
        edges_file_path (str): Path to the file containing edges.

    Returns:
        np.ndarray: A NumPy array representing the adjacency matrix of the graph.

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

    # Initialize a (max_vertex + 1) x (max_vertex + 1) matrix with zeros using NumPy
    adjacency_matrix = np.zeros((max_vertex + 1, max_vertex + 1), dtype=int)

    for u, v in edges:
        adjacency_matrix[u][v] = 1
        adjacency_matrix[v][u] = 1

    return adjacency_matrix


def get_top_centrality(centrality, top_n: int=DEFLAULT_NODES) -> list:
    """
    Get the top N nodes based on their centrality scores.

    Args:

        centrality (Union[dict, np.ndarray]): Dictionary or array of centrality scores.
        top_n (int): Number of top nodes to return.

    Returns:
        list: List of tuples containing the node and its centrality score.
    
    Raises:
        ValueError: If top_n is not positive.
        TypeError: If top_n is not an integer.
    """
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if not isinstance(top_n, int):
        raise TypeError("top_n must be an integer")
    
    if isinstance(centrality, dict):
        # Handle dictionary input
        sorted_centrality = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
        return sorted_centrality[:top_n]
    elif isinstance(centrality, np.ndarray):
        # Convert array to (node_id, centrality_score) tuples
        centrality_tuples = [(i, score) for i, score in enumerate(centrality)]
        sorted_centrality = sorted(centrality_tuples, key=lambda item: item[1], reverse=True)
        return sorted_centrality[:top_n]
    else:
        raise TypeError("centrality must be a dictionary or numpy array")

def compare_centrality_with_egos(centrality_list: list[tuple], ego_vertices: set) -> None:
    """
    Compare centrality scores with ego vertices.

    Args:
        centrality_dict (list[tuple]): List of tuples containing nodes and their centrality scores.
        ego_file_path (str): Path to the file containing ego vertices.

    Returns:
        dict: Dictionary with ego vertices and their centrality scores.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid data.
    """
    centrality_nodes = set(node for node, _ in centrality_list)
    correct_ego_vertices = ego_vertices.intersection(centrality_nodes)
    incorrect_ego_vertices = centrality_nodes - correct_ego_vertices
    missed_ego_vertices = ego_vertices - correct_ego_vertices
    print("Correct ego vertices:", correct_ego_vertices)
    if incorrect_ego_vertices:
        print("Incorrect prediction of top centrality vertices:", incorrect_ego_vertices)
    if missed_ego_vertices:
        print("Missed ego vertices:", missed_ego_vertices)


def plot_social_network_with_centrality(
    adjacency_list,
    centrality,
    centrality_measure,
    top_nodes,
    layout_algorithm=DEFAULT_LAYOUT,
):
    """
    Visualizes a social network using NetworkX and Matplotlib.

    Args:
        adjacency_list (dict): Adjacency list data as a dictionary of adjacency_list.
        centrality (dict): Centrality measure to visualize ('betweenness', 'eigenvector', 'pagerank').
        centrality_measure (str): The centrality measure used for visualization.
        top_nodes (list): List of top N nodes to highlight.
        layout_algorithm (str): Layout algorithm ('spring', 'circular', 'kamada_kawai').
    """
    # Load adjacency list
    if not isinstance(adjacency_list, dict):
        raise TypeError("adjacency_list must be a dictionary")

    # Create NetworkX graph from adjacency list
    nx_graph = nx.Graph(adjacency_list)
    
    # Normalize centrality values for visualization
    min_centrality = min(centrality.values())
    max_centrality = max(centrality.values())
    norm = Normalize(vmin=min_centrality, vmax=max_centrality)
    
    # Set node sizes based on normalized centrality
    node_sizes = [norm(centrality[node]) * NODE_SIZE for node in nx_graph.nodes()]
    
    # Create set of top nodes for highlighting
    top_nodes_set = set(node for node, _ in top_nodes)
    
    # Choose layout based on specified algorithm
    if layout_algorithm == "spring":
        pos = nx.spring_layout(nx_graph, seed=POSITION_SEED)
    elif layout_algorithm == "circular":
        pos = nx.circular_layout(nx_graph)
    elif layout_algorithm == "kamada_kawai":
        pos = nx.kamada_kawai_layout(nx_graph)
    else:
        raise ValueError("Invalid layout_algorithm. Choose 'spring', 'circular', or 'kamada_kawai'.")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    
    # Draw all nodes with color based on centrality
    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        node_size=node_sizes,
        node_color=[centrality[node] for node in nx_graph.nodes()],
        cmap=COLOR_MAP,
        alpha=TRANSPARENCY,
        ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        nx_graph,
        pos,
        width=EDGE_WIDTH,
        edge_color=EDGE_COLOR,
        alpha=TRANSPARENCY,
        ax=ax
    )
    
    # Draw the top nodes with a different border color to highlight them
    if top_nodes_set:
        top_node_sizes = [norm(centrality[node]) * HIGHLIGHT_NODE_SIZE for node in top_nodes_set]
        nx.draw_networkx_nodes(
            nx_graph,
            pos,
            nodelist=list(top_nodes_set),
            node_size=top_node_sizes,
            node_color=[centrality[node] for node in top_nodes_set],
            cmap=COLOR_MAP,
            edgecolors=NODE_EDGE_COLOR,
            linewidths=EDGE_WIDTH,
            alpha=TRANSPARENCY,
            ax=ax
        )
    
    # Add node labels for top nodes only to prevent overcrowding
    labels = {node: str(node) for node in top_nodes_set}
    nx.draw_networkx_labels(
        nx_graph,
        pos,
        labels=labels,
        font_size=LABEL_SIZE,
        font_color=LABEL_FONT_COLOR,
        font_weight='bold',
        ax=ax
    )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=COLOR_MAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f'{centrality_measure.capitalize()} Centrality')
    
    # Set title and remove axis
    plt.title(f"Social Network Visualization - {centrality_measure.capitalize()} Centrality")
    plt.axis('off')
    
    #Add legend for top influencers
    legend_elements = [plt.Line2D([0], [0], marker='o', color=LEGEND_COLOR, 
                                 markerfacecolor=LABEL_FONT_COLOR, markersize=LABEL_SIZE, 
                                 label=f'Top {len(top_nodes_set)} Influencers')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save the figure
    output_filename = f"{centrality_measure.title()} Graph.png"
    plt.savefig(GARPH_PATH + output_filename, bbox_inches='tight')
    print(f"Social network visualization saved as {output_filename}")


def plot_social_network(adjacency_list:dict, ego_nodes:set) -> None:
    """
    Visualizes a social network using NetworkX and Matplotlib.

    Args:
        adjacency_list (dict): Adjacency list data as a dictionary of adjacency_list.
        ego_nodes (set): Set of ego nodes to highlight.

    Returns:
        None
    """
    # Load adjacency list
    if not isinstance(adjacency_list, dict):
        raise TypeError("adjacency_list must be a dictionary")

    # Create NetworkX graph from adjacency list
    nx_graph = nx.Graph(adjacency_list)
    
    # Choose layout based on specified algorithm
    pos = nx.spring_layout(nx_graph, seed=POSITION_SEED)

    # Create figure and axis
    _, ax = plt.subplots(figsize=(FIGURE_SIZE, FIGURE_SIZE))

    # Draw all nodes with color based on centrality
    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        node_color=NODE_COLOR,
        node_size=SAMLL_NODE_SIZE,
        alpha=TRANSPARENCY,
        ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        nx_graph,
        pos,
        width=EDGE_WIDTH,
        edge_color=EDGE_COLOR,
        alpha=TRANSPARENCY,
        ax=ax
    )
    
    # Highlight ego nodes with a different node color
    ego_node_size = [HIGHLIGHT_NODE_SIZE for _ in ego_nodes]
    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        nodelist=list(ego_nodes),
        node_size=ego_node_size,
        node_color=EGO_NODE_COLOR,
        edgecolors=NODE_EDGE_COLOR,
        linewidths=EDGE_WIDTH,
        alpha=TRANSPARENCY,
        ax=ax
    )

    # Add node labels for ego nodes only to prevent overcrowding
    labels = {node: str(node) for node in ego_nodes}
    nx.draw_networkx_labels(
        nx_graph,
        pos,
        labels=labels,
        font_size=LABEL_SIZE,
        font_color=EGO_NODE_LABEL_COLOR,
        font_weight='bold',
        ax=ax
    )

    # Add legend for ego nodes
    legend_elements = [plt.Line2D([0], [0], marker='o', color=LEGEND_COLOR, 
                                 markerfacecolor=EGO_NODE_COLOR, markersize=LABEL_SIZE, 
                                 label=f'Ego Nodes')]
    ax.legend(handles=legend_elements, loc='upper right')

    # Set title and remove axis
    plt.title("SNAP Facebook Social Network Visualization")
    plt.axis('off')

    # Save the figure
    output_filename = "Facebook Dataset.png"
    plt.savefig(GARPH_PATH + output_filename, bbox_inches='tight')
    print(f"Social network visualization saved as {output_filename}")