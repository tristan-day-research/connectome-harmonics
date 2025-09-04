import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
import pygsp


def compute_graph_laplacian_eigen(data_dict):
    result_dict = {}

    for subject, matrix in data_dict.items():
        # Compute the graph Laplacian
        L = laplacian(matrix, normed=True)

        # Compute all eigenvectors and eigenvalues
        eigenvalues, eigenvectors = eigh(L)

        result_dict[subject] = {
            'eigenvectors': eigenvectors,
            'eigenvalues': eigenvalues
        }

    return result_dict


def compute_connectome_harmonics(adjacency_dict, lap_type="normalized"):
    """
    Returns a dictionary where each key is a subject number and each value is another dictionary with 
    keys 'eigenvectors' and 'eigenvalues' corresponding to the eigenvectors and eigenvalues of the graph Laplacian
    for the adjacency matrix associated with that subject.
    
    Parameters:
    - adjacency_dict: a dictionary where each key is a subject number and each value is a 376x376 connectivity matrix.
    - lap_type: either "combinatorial" or "normalized"
    """
    
    result_dict = {}
    
    for subject, adjacency_matrix in adjacency_dict.items():
        # Ensure the diagonal is zero as PyGSP does not support self-loops
        np.fill_diagonal(adjacency_matrix, 0)
        
        # Create a PyGSP graph from the adjacency matrix
        G_fd = pygsp.graphs.Graph(adjacency_matrix)
        
        # Compute the Laplacian of the graph
        G_fd.compute_laplacian(lap_type=lap_type)
        
        # Compute the Fourier basis of the graph, which gives us the eigenvectors and eigenvalues
        G_fd.compute_fourier_basis()
        
        # Store the eigenvectors and eigenvalues in the result dictionary
        result_dict[subject] = {
            'eigenvectors': G_fd.U,
            'eigenvalues': G_fd.e
        }
    
    return result_dict

# Usage:
# Assuming adjacency_dict is your dictionary of adjacency matrices
# harmonics_dict = compute_harmonics(adjacency_dict)

# This function is for a numpy array rather than a dict

def compute_laplacian_eigendecomposition(adjacency_matrix, lap_type="normalized"):
    """
    Compute eigenvalues and eigenvectors of the graph Laplacian of a single adjacency matrix.
    
    Parameters:
    - adjacency_matrix: 2D NumPy array (square)
    - lap_type: 'normalized' or 'combinatorial'
    
    Returns:
    - eigenvalues: 1D NumPy array
    - eigenvectors: 2D NumPy array (columns are eigenvectors)
    """
    # Enforce symmetry
    adjacency_matrix = 0.5 * (adjacency_matrix + adjacency_matrix.T)
    
    # Zero diagonal (no self-loops)
    np.fill_diagonal(adjacency_matrix, 0)
    
    normed = (lap_type == "normalized")
    L = laplacian(adjacency_matrix, normed=normed)
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = eigh(L)
    
    return eigenvalues, eigenvectors