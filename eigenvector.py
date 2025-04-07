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
    - centrality (numpy.ndarray): Centrality score for each node.
    """
    n = matrix.shape[0]

    # Start with a non-uniform vector to break symmetry
    # Using random initialization is better than uniform values
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)  # Normalize the initial vector

    for _ in range(max_iter):
        x_new = matrix @ x

        # Check if the vector is close to zero
        norm = np.linalg.norm(x_new)

        if norm < NORM_THRESHOLD:
            # Return zero centrality if the norm is too small
            return np.zeros(n)

        x_new = x_new / norm  # Normalize

        # Check for convergence
        if np.linalg.norm(x - x_new) < tol:
            break

        x = x_new

    # Ensure the results are positive (convention for eigenvector centrality)
    if np.any(x < 0):
        x = -x

    # For highly symmetric cases (like complete graphs), ensure more numerical stability
    # by enforcing theoretical equality where appropriate
    if np.allclose(matrix, matrix.T):  # Symmetric matrix
        # Check if matrix is special case like simple path, cycle, or complete graph
        row_sums = np.sum(matrix, axis=1)
        if np.allclose(row_sums, row_sums[0]):  # All nodes have same degree
            # This includes cases like complete graphs, cycles, etc.
            return np.ones(n) / np.sqrt(n)  # Equal centrality for all nodes

    return x
