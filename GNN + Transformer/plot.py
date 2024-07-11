import networkx as nx
import matplotlib.pyplot as plt

def plot_tsp_graph(coordinates, distances, tour=None):
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
    
    if tour is not None:
        # Draw tour without the last leg
        if isinstance(tour, list) and all(isinstance(item, int) for item in tour):
            tour_edges = [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)]
            nx.draw_networkx_edges(G, pos, edgelist=tour_edges, edge_color='red', width=2)
        else:
            raise ValueError("Tour must be a list of integers.")
    
    plt.title('TSP Graph')
    plt.show()
