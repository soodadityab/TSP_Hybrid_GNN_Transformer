from data import generate_tsp_instances
from train_gem import train_model
from plot import plot_tsp_graph
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def decode_tsp_tour(prob_matrix, num_cities):
  """
  Decodes the output probability matrix into a TSP tour ensuring no node is visited twice
  and the nodes are within the valid range. Also calculates the negative log-likelihood of the tour.
  """
  tour = []
  visited = set()
  log_likelihood = 0.0
  current_node = torch.argmax(prob_matrix[0]).item()
  tour.append(current_node)
  visited.add(current_node)

  for i in range(num_cities - 1):
    # Mask visited nodes by setting their probability to -inf
    prob_matrix[:, list(visited)] = -float('inf')  
    next_node = torch.argmax(prob_matrix[current_node]).item()
    
    # Check if the current_node is valid
    if next_node < 0 or next_node >= num_cities:
      raise ValueError(f"Decoded node index {next_node} is out of bounds.")
    
    # Update log-likelihood
    log_likelihood += torch.log(prob_matrix[current_node, next_node]).item()
    
    tour.append(next_node)
    visited.add(next_node)
    current_node = next_node

  return tour, log_likelihood

def calculate_total_distance(tour, coordinates):
    total_distance = 0.0
    num_cities = len(tour)
    for i in range(num_cities):
        start_city = coordinates[tour[i]]
        end_city = coordinates[tour[(i + 1) % num_cities]]
        distance = np.linalg.norm(np.array(start_city) - np.array(end_city))
        print(f"Distance from {start_city} to {end_city}: {distance}")
        total_distance += distance
    return total_distance

# Generate TSP Instances
num_instances = 5000
num_cities = 10
tsp_instances = generate_tsp_instances(num_instances, num_cities)

print("Created TSP instances, starting training")

# Train Hybrid Model
num_epochs = 11
trained_model = train_model(num_epochs, tsp_instances, num_cities)

print("Training complete")

# Plot the result
coordinates, distances = tsp_instances[0]
edge_index = torch.tensor(np.array(np.meshgrid(range(len(coordinates)), range(len(coordinates)))).reshape(2, -1), dtype=torch.long).to(device)
x = torch.tensor(coordinates, dtype=torch.float).to(device)
data = Data(x=x, edge_index=edge_index).to(device)

# Get model output
output = trained_model(data)

print(output.shape)  # Should be [num_cities, num_cities]

# Decode the TSP tour
predicted_tour = decode_tsp_tour(output, num_cities)
predicted_tour = predicted_tour[0]
print(f"Predicted tour: {predicted_tour}")

# Plot the TSP graph with the predicted tour
plot_tsp_graph(coordinates, distances, predicted_tour)
