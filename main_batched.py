from data import generate_tsp_instances
from train_batched import train_model
from plot import plot_tsp_graph
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import torch
import numpy as np
from utils import decode_tsp_tour, calculate_total_distance
from data import solve_tsp_exact

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate TSP Instances
num_instances = 1000
num_cities = 30
batch_size = 1
tsp_instances = generate_tsp_instances(num_instances, num_cities)

print("Created TSP instances, starting training")

# Train Hybrid Model
num_epochs = 15
trained_model = train_model(num_epochs, tsp_instances, num_cities, batch_size)

print("Training complete")
trained_model.eval()

# Plot the result
coordinates, distances = tsp_instances[0]
edge_index = np.array(np.meshgrid(range(len(coordinates)), range(len(coordinates)))).reshape(2, -1)
mask = edge_index[0] != edge_index[1]
edge_index = edge_index[:, mask]
edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

x = torch.tensor(coordinates, dtype=torch.float).to(device)
distances_tensor = torch.tensor(distances, dtype=torch.float).to(device)
data = Data(x=x, edge_index=edge_index).to(device)

# Convert single instance to batch
data_list = [data]
batch_data = Batch.from_data_list(data_list).to(device)

# Get model output
output = trained_model(batch_data)

# Decode the TSP tour
predicted_tour = decode_tsp_tour(output[0], num_cities)
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
