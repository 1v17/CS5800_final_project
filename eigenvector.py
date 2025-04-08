import numpy as np

ITERATION = 100
TORLERANCE = 1e-6

def eigenvector_centrality(matrix, max_iter=ITERATION, tol=TORLERANCE):
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
    
    for _ in range(max_iter):
        new_centrality = matrix @ centrality  # Matrix-vector multiplication
        new_centrality = new_centrality / np.linalg.norm(new_centrality)  # Normalize
        
        # Check for convergence
        if np.linalg.norm(new_centrality - centrality) < tol:
            break
            
        centrality = new_centrality
    
    # Normalize the centrality scores
    # Convert the numpy array to a dictionary
    return {i: float(score) for i, score in enumerate(centrality)}
