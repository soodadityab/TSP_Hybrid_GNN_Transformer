from data import generate_tsp_instances, solve_tsp_exact
from unbatched.train_gem import train_model
from plot import plot_tsp_graph
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
import numpy as np
from utils import decode_tsp_tour, calculate_total_distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate TSP Instances
num_instances = 100
num_cities = 30
tsp_instances = generate_tsp_instances(num_instances, num_cities)

print("Created TSP instances, starting training")

# Train Hybrid Model
num_epochs = 100
trained_model = train_model(num_epochs, tsp_instances, num_cities)

print("Training complete")
trained_model.eval()

# Plot the result
coordinates, distances = tsp_instances[0]
edge_index = torch.tensor(np.array(np.meshgrid(range(len(coordinates)), range(len(coordinates)))).reshape(2, -1), dtype=torch.long).to(device)
x = torch.tensor(coordinates, dtype=torch.float).to(device)
data = Data(x=x, edge_index=edge_index).to(device)

# Get model output
output = trained_model(data)

# print(output.shape)  # Should be [num_cities, num_cities]

# Decode the TSP tour
predicted_tour = decode_tsp_tour(output, num_cities)
predicted_tour = predicted_tour[0]
print(f"Predicted tour: {predicted_tour}")
total_distance = calculate_total_distance(predicted_tour, coordinates)
print(f"Total distance of the predicted tour: {total_distance}")
# Plot the TSP graph with the predicted tour
plot_tsp_graph(coordinates, distances, predicted_tour)

optimalT, optimalD = solve_tsp_exact(distances)
plot_tsp_graph(coordinates, distances, optimalT)
print(f"distance optimal: {optimalD}")
print(f"tour optimal: {optimalT}")
