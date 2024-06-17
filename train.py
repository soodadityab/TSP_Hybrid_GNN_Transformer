import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy as np
from model import GNNEmbeds, TransformerTSP, Hybrid

# Ensure the model and data are on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def save_model(model, path):
    torch.save(model.state_dict(), path)

save_model_path = 'trained_tsp_model.pth'

# LOSS FUNCTION DISTANCE

def calculate_total_distance(tour, coordinates):
    total_distance = 0.0
    num_cities = len(tour)
    for i in range(num_cities):
        start_city = coordinates[tour[i]]
        end_city = coordinates[tour[(i + 1) % num_cities]]
        distance = np.linalg.norm(np.array(start_city) - np.array(end_city))
        total_distance += distance
    return total_distance


def tsp_loss(output, distances, coordinates):
    tour = []
    visited = set()
    current_node = torch.argmax(output[0]).item()
    tour.append(current_node)
    visited.add(current_node)

    num_cities = output.size()[0]
    for _ in range(num_cities - 1):
        output[:, list(visited)] = -float('inf')  # Mask visited nodes by setting their probability to -inf
        current_node = torch.argmax(output[current_node]).item()
        
        # Check if the current_node is valid
        if current_node < 0 or current_node >= num_cities:
            raise ValueError(f"Decoded node index {current_node} is out of bounds.")
        
        if current_node not in visited:
            tour.append(current_node)
            visited.add(current_node)
    
    distLoss = torch.tensor(calculate_total_distance(tour, coordinates), requires_grad=True)
    return distLoss


def train_model(num_epochs, tsp_instances, num_cities):
    gnn = GNNEmbeds(in_channels=2, out_channels=128).to(device)
    transformer = TransformerTSP(hidden_dim=128, num_heads=8, num_layers=6).to(device)
    model = Hybrid(gnn, transformer, num_cities).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for coordinates, distances in tsp_instances:
            # Create edge index for a complete graph

            # Create edge index for a complete graph without self-loops
            edge_index = np.array(np.meshgrid(range(num_cities), range(num_cities))).reshape(2, -1)
            # Filter out self-loops
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            # Convert to PyTorch tensor and move to the appropriate device
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

            x = torch.tensor(coordinates, dtype=torch.float).to(device)
            distances_tensor = torch.tensor(distances, dtype=torch.float).to(device)
            data = Data(x=x, edge_index=edge_index).to(device)
            
            optimizer.zero_grad()
            output = model(data)

            loss = tsp_loss(output, distances_tensor, coordinates)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(tsp_instances)}')

    save_model(model, save_model_path)
    print(f'Model saved to {save_model_path}')
    return model
