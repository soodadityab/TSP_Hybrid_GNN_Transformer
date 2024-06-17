import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Coordinates of the tour
coordinates = [
    [0.65971964, 0.43021783],
    [0.07667023, 0.61731952],
    [0.60741434, 0.87261335],
    [0.29240596, 0.78020028],
    [0.64242792, 0.40841993],
    [0.15469572, 0.23753343],
    [0.21442187, 0.11450249],
    [0.66315646, 0.47261216],
    [0.8668531, 0.86759692],
    [0.66022948, 0.9751236 ]
]

# Predicted tour
predicted_tour = [7, 9, 0, 4, 2, 3, 5, 8, 1, 6]

# True optimal tour
optimal_tour = [0, 4, 6, 5, 1, 3, 2, 9, 8, 7, 0]

# Distance matrix
distances = np.zeros((len(coordinates), len(coordinates)))

# Define distances as provided
distances[0][1] = 0.6123346032281769
distances[1][2] = 0.5889518273313963
distances[2][3] = 0.32828411268497976
distances[3][4] = 0.5106231505714329
distances[4][5] = 0.5168025622156699
distances[5][6] = 0.13676192375446833
distances[6][7] = 0.5741125873659381
distances[7][8] = 0.44441566801444504
distances[8][9] = 0.23292768842811154
distances[9][0] = 0.544906011959432

# Since the distance matrix is symmetric, fill the other half
distances = distances + distances.T

def plot_tsp_graph(coordinates, distances, predicted_tour=None, optimal_tour=None):
    G = nx.Graph()
    
    # Add nodes
    for i, (x, y) in enumerate(coordinates):
        G.add_node(i, pos=(x, y))
    
    # Add edges with weights
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            G.add_edge(i, j, weight=distances[i][j])
    
    pos = nx.get_node_attributes(G, 'pos')
    
    plt.figure(figsize=(10, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='blue')
    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', alpha=0.5)
    # Draw labels
    nx.draw_networkx_labels(G, pos)
    
    if predicted_tour is not None:
        # Draw predicted tour in red
        if isinstance(predicted_tour, list) and all(isinstance(item, int) for item in predicted_tour):
            predicted_edges = [(predicted_tour[i], predicted_tour[i + 1]) for i in range(len(predicted_tour) - 1)]
            predicted_edges.append((predicted_tour[-1], predicted_tour[0]))  # Close the tour
            nx.draw_networkx_edges(G, pos, edgelist=predicted_edges, edge_color='red', width=2, label='Predicted Tour')
        else:
            raise ValueError("Predicted tour must be a list of integers.")
    
    if optimal_tour is not None:
        # Draw optimal tour in blue
        if isinstance(optimal_tour, list) and all(isinstance(item, int) for item in optimal_tour):
            optimal_edges = [(optimal_tour[i], optimal_tour[i + 1]) for i in range(len(optimal_tour) - 1)]
            optimal_edges.append((optimal_tour[-1], optimal_tour[0]))  # Close the tour
            nx.draw_networkx_edges(G, pos, edgelist=optimal_edges, edge_color='blue', width=2, style='dashed', label='Optimal Tour')
        else:
            raise ValueError("Optimal tour must be a list of integers.")
    
    plt.title('TSP Graph')
    plt.legend()
    plt.show()

# Call the function with coordinates, distances, predicted tour, and optimal tour
plot_tsp_graph(coordinates, distances, predicted_tour, optimal_tour)
