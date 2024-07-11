from data import generate_tsp_instance
from plot import plot_tsp_graph

# Generate a TSP instance
num_cities = 20
coordinates, distances = generate_tsp_instance(num_cities)

# Plot the TSP graph
plot_tsp_graph(coordinates, distances)