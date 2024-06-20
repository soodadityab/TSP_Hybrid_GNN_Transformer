import torch
import numpy as np
import pickle
from unbatched.model import GNNEmbeds, TransformerTSP, Hybrid
from torch_geometric.data import Data
from utils import decode_tsp_tour, calculate_total_distance
from plot import plot_tsp_graph

NUM_CITIES = 30
VALIDATION_SIZE = 2000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = 'trained_tsp_model.pth'
gnn = GNNEmbeds(in_channels=2, out_channels=128).to(device)
transformer = TransformerTSP(hidden_dim=128, num_heads=8, num_layers=6).to(device)
model = Hybrid(gnn, transformer, NUM_CITIES).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

with open('validation_set.pkl', 'rb') as f:
  validation_set = pickle.load(f)

total_model_distance = 0
total_nn_distance = 0
total_percentage_deviation = 0
model_beats_optimal = 0
model_is_optimal = 0

for coordinates, distances, optimal_tour, optimal_distance in validation_set:
  edge_index = np.array(np.meshgrid(range(NUM_CITIES), range(NUM_CITIES))).reshape(2, -1)
  mask = edge_index[0] != edge_index[1]
  edge_index = edge_index[:, mask]
  # Convert to PyTorch tensor and move to the appropriate device
  edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
  x = torch.tensor(coordinates, dtype=torch.float).to(device)
  distances_tensor = torch.tensor(distances, dtype=torch.float).to(device)
  data = Data(x=x, edge_index=edge_index).to(device)
  output = model(data)

  predicted_tour = decode_tsp_tour(output, NUM_CITIES)
  predicted_tour = predicted_tour[0]

  # # Plot the TSP graph with the predicted tour
  # plot_tsp_graph(coordinates, distances, predicted_tour)

  # Calculate predicted tour distance based on your model's output and distances
  predicted_distance = calculate_total_distance(predicted_tour, coordinates)
  print(predicted_distance)

  # Calculate nearest neighbor tour distance (already available in validation set)
  nn_tour, nn_distance = optimal_tour, optimal_distance  # Extract tour and distance from validation set
  # plot_tsp_graph(coordinates, distances, nn_tour)
  
  # metrics
  if predicted_distance < optimal_distance:
    model_beats_optimal += 1
  elif predicted_distance == optimal_distance:
    model_is_optimal +=1

  percentage_deviation = ((predicted_distance - optimal_distance) / optimal_distance) * 100
  total_percentage_deviation += percentage_deviation

  total_model_distance += predicted_distance
  print(f"total is {total_model_distance}")
  total_nn_distance += nn_distance

print(f"The total distance from model predictions is {total_model_distance:.2f}")
print(f"The total distance from NN predictions is {total_nn_distance:.2f}")

print(f"On average, the distance predicted by the model is {total_model_distance/VALIDATION_SIZE:.2f}")
print(f"On average, the distance predicted by NN is {total_nn_distance/VALIDATION_SIZE:.2f}")

print(f"Average percentage deviation from optimal for the model: {total_percentage_deviation/VALIDATION_SIZE:.2f}%")

print(f"Model was optimal {model_is_optimal} times; model beat optimal {model_beats_optimal} times")