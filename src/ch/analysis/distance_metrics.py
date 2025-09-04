import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats


def calculate_edge_weight_sum(G, harmonic_a, harmonic_b):
    inter_subgraph_edge_weight_sum = 0

    for node1 in harmonic_a:
        for node2 in harmonic_b:
            if G.has_edge(node1, node2):
                inter_subgraph_edge_weight_sum += G[node1][node2]['weight']

    return inter_subgraph_edge_weight_sum


def inter_subgraph_edge_weight_sum(active_nodes_index_list, adjacency_matrix, analysis_params):
    # Unpack analysis_params dictionary 
    
    harmonics_to_analyze = analysis_params['harmonics_to_analyze']

    num_harmonics = max(harmonics_to_analyze)
   

    inter_harmonic_connectivity_matrix = np.zeros([num_harmonics, num_harmonics])

    G = nx.from_numpy_array(adjacency_matrix)    # Create a graph from the weighted adjacency matrix

    for i in range(num_harmonics):
        for j in range(num_harmonics):
            harmonic_a, harmonic_b = active_nodes_index_list[i], active_nodes_index_list[j]
            
            inter_subgraph_edge_weight_sum = calculate_edge_weight_sum(G, harmonic_a, harmonic_b)
            
            # print(i, j, inter_subgraph_edge_weight_sum)
            inter_harmonic_connectivity_matrix[i][j] = inter_subgraph_edge_weight_sum

    return inter_harmonic_connectivity_matrix


def average_shortest_path(active_nodes_index_list, adjacency_matrix, NUM_HARMONICS):

    inter_harmonic_connectivity_matrix = np.zeros([NUM_HARMONICS, NUM_HARMONICS])  
    G = nx.from_numpy_array(adjacency_matrix)    # Create a graph from the weighted adjacency matrix

    shortest_path_lengths = []

    for i in range(NUM_HARMONICS):

        print("Calculating for harmonic", i)

        for j in range(NUM_HARMONICS):
            harmonic_a, harmonic_b = active_nodes_index_list[i], active_nodes_index_list[j]
            
            # inter_subgraph_edge_weight_sum = calculate__edge_weight_sum(G, harmonic_a, harmonic_b)
            
            # print(i, j, inter_subgraph_edge_weight_sum)
            # inter_harmonic_connectivity_matrix[i][j] = inter_subgraph_edge_weight_sum

            for node1 in harmonic_a:
                for node2 in harmonic_b:
                    shortest_path_length = nx.shortest_path_length(G, node1, node2, weight='weight')
                    shortest_path_lengths.append(shortest_path_length)

            average_shortest_path_length = sum(shortest_path_lengths) / len(shortest_path_lengths)
            inter_harmonic_connectivity_matrix[i][j] = average_shortest_path_length

    return inter_harmonic_connectivity_matrix