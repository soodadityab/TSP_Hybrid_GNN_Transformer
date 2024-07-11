from data import generate_tsp_instances_validation
from train_batched import train_model
from plot import plot_tsp_graph
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import torch
import numpy as np
from utils import decode_tsp_tour, calculate_total_distance
from data import solve_tsp_exact, create_batches

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate TSP Instances
num_instances = 1
num_cities = 10
batch_size = 32
tsp_instances = generate_tsp_instances_validation(num_instances, num_cities)
print("Created TSP instances, starting training")

# Train Hybrid Model
num_epochs = 11
trained_model = train_model(num_epochs, tsp_instances, num_cities, batch_size)

# Define function to decode the model's output and plot the paths
def visualize_model_vs_optimal(model, tsp_instance):
    coordinates, distances, optimal_tour, optimal_distance = tsp_instance

    edge_index = np.array(np.meshgrid(range(num_cities), range(num_cities))).reshape(2, -1)
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

    x = torch.tensor(coordinates, dtype=torch.float).to(device)
    row, col = edge_index
    edge_attr = distances[row, col].reshape(-1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device).unsqueeze(-1)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)
    data = [data]
    batch_data = Batch.from_data_list(data).to(device)

    # model.eval()
    with torch.no_grad():
        output = model(batch_data, optimal_tour)
    print(output)
    predicted_tour = decode_tsp_tour(output, num_cities)
    print(calculate_total_distance(predicted_tour, coordinates))
    print(f"predicted: {predicted_tour}")
    print(f"optimal: {optimal_distance}")
    
    # Plot optimal and model's predicted tours
    plot_tsp_graph(coordinates, distances, optimal_tour)
    plot_tsp_graph(coordinates, distances, predicted_tour)

# Select an instance from the validation set for visualization
validation_instance = tsp_instances[0]
visualize_model_vs_optimal(trained_model, validation_instance)
