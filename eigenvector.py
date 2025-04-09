import numpy as np

NORM_THRESHOLD = 1e-10
DEFLAUT_ITERATIONS = 100
TOLERANCE = 1e-6


def eigenvector_centrality(matrix, max_iter=DEFLAUT_ITERATIONS, tol=TOLERANCE):
    """
    Calculate the eigenvector centrality of a graph given by its adjacency matrix.
    
    Parameters:
    - matrix (numpy.ndarray): Square adjacency matrix.
    - max_iter (int): Maximum number of iterations.
    - tol (float): Convergence tolerance.
    
    Returns:
    - centrality (dict): Dictionary mapping node indices to centrality scores.
    """
    n = matrix.shape[0]
    centrality = np.ones(n)  # Initialize with all ones
    
    iteration = 0
    tolerance = np.inf
    while iteration < max_iter and tolerance > tol:
        new_centrality = matrix @ centrality  # Matrix-vector multiplication
        new_centrality = new_centrality / np.linalg.norm(new_centrality)  # Normalize
        
        # Update tolerance and centrality
        tolerance = np.linalg.norm(new_centrality - centrality)           
        centrality = new_centrality
        iteration += 1
    
    # Convert the numpy array to a dictionary
    return {i: float(score) for i, score in enumerate(centrality)}
