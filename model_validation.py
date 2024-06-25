import torch
import numpy as np
import pickle
from model_AR import GNNEmbeds, TransformerTSP, Hybrid
from torch_geometric.data import Data, Batch
from utils import decode_tsp_tour, calculate_total_distance
from plot import plot_tsp_graph

NUM_CITIES = 10
VALIDATION_SIZE = 200
BATCH_SIZE = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = 'trained_tsp_model.pth'

# Initialize and load the trained model
gnn = GNNEmbeds(in_channels=2, out_channels=128).to(device)
transformer = TransformerTSP(hidden_dim=128, num_heads=8, num_layers=6, max_seq_length=NUM_CITIES).to(device)
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

# Validation loop
for instance_idx, instance in enumerate(validation_set):
    coordinates, distances, optimal_tour, optimal_distance = instance

    edge_index = np.array(np.meshgrid(range(NUM_CITIES), range(NUM_CITIES))).reshape(2, -1)
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

    x = torch.tensor(coordinates, dtype=torch.float).to(device)
    row, col = edge_index
    edge_attr = distances[row, col].reshape(-1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device).unsqueeze(-1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)
    data = Batch.from_data_list([data]).to(device)

    with torch.no_grad():
        output = model(data, [optimal_tour])

    predicted_tour = decode_tsp_tour(output, NUM_CITIES)
    model_distance = calculate_total_distance(predicted_tour, coordinates)

    total_model_distance += model_distance
    total_nn_distance += optimal_distance

    # Check if model beats optimal or is optimal
    if model_distance < optimal_distance:
        model_beats_optimal += 1
        plot_tsp_graph(coordinates, distances, optimal_tour)
        print(f"optimal distance: {optimal_distance}")
        plot_tsp_graph(coordinates, distances, predicted_tour)
        print(f"model distance: {model_distance}")
    if model_distance == optimal_distance:
        model_is_optimal += 1
    else:
        percentage_deviation = ((model_distance - optimal_distance) / optimal_distance) * 100
        total_percentage_deviation += percentage_deviation


    # Optionally, plot the graphs for visual validation
    # if instance_idx < 5:  # Plot only first 5 instances
    #     print(f"Instance {instance_idx + 1}:")
    #     plot_tsp_graph(coordinates, distances, optimal_tour)
    #     plot_tsp_graph(coordinates, distances, predicted_tour)

# Calculate averages
avg_model_distance = total_model_distance / VALIDATION_SIZE
avg_nn_distance = total_nn_distance / VALIDATION_SIZE
avg_percentage_deviation = total_percentage_deviation / VALIDATION_SIZE

# Print results
print(f"Avg Model Distance: {avg_model_distance}")
print(f"Avg Optimal NN Distance: {avg_nn_distance}")
print(f"Avg Percentage Deviation: {avg_percentage_deviation}%")
print(f"Number of instances where Model Beats Optimal: {model_beats_optimal}")
print(f"Number of instances where Model is Optimal: {model_is_optimal}")
