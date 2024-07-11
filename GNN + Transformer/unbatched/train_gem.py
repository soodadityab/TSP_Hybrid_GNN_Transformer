import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy as np
from unbatched.model import GNNEmbeds, TransformerTSP, Hybrid
import torch.optim.lr_scheduler as lr_scheduler
# from data import solve_tsp_exact


# Ensure the model and data are on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def save_model(model, path):
    torch.save(model.state_dict(), path)

save_model_path = 'trained_tsp_model.pth'

def normalize_coordinates(coordinates):
    coordinates = np.array(coordinates)
    min_val = coordinates.min(axis=0)
    max_val = coordinates.max(axis=0)
    normalized = (coordinates - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero
    return normalized


# def tsp_loss(output, distances):
#     """
#     Calculates the loss function incorporating total distance and penalties for revisiting or missing nodes.
    
#     Args:
#     - output (torch.Tensor): The output probabilities from the model with shape (num_nodes, num_nodes).
#     - distances (torch.Tensor): A matrix containing the distances between nodes with shape (num_nodes, num_nodes).
    
#     Returns:
#     - total_loss (torch.Tensor): The combined loss.
#     """
#     num_nodes = output.shape[0]
    
#     # Get the path from the model output
#     _, path_indices = torch.max(output, dim=-1)
#     # print(f"path indices: {path_indices}")
    
#     # Calculate the total distance for the path
#     total_distance = torch.tensor(0.0, requires_grad=True)
#     for i in range(num_nodes - 1):
#         total_distance = total_distance + distances[path_indices[i], path_indices[i + 1]]

#     # print(f"tot dist: {total_distance}")

#     # Penalize revisiting nodes or missing nodes
#     node_visits = torch.zeros(num_nodes, device=output.device)
#     for i in range(num_nodes):
#         node_visits = node_visits + torch.nn.functional.one_hot(path_indices[i], num_classes=num_nodes).float()
#     # print(f"node visits: {node_visits}")
    
#     revisit_penalty = torch.sum((node_visits - 1).clamp(min=0))  # Penalize revisits
#     missing_penalty = torch.sum((1 - node_visits).clamp(min=0))  # Penalize missing nodes
#     # print(f"revisit penalty: {revisit_penalty} and missing_penalty: {missing_penalty}")
    
#     # Combine the distance loss and the penalties
#     alpha = 1.0  # Weight for distance
#     beta = 5.0  # Weight for revisit penalty
#     gamma = 5.0  # Weight for missing node penalty
    
#     total_loss = alpha * total_distance + beta * revisit_penalty + gamma * missing_penalty
    
#     return total_loss



# def tsp_loss(output, distances):
#   """
#   Calculates the loss function incorporating total distance and negative log-likelihood
#   to discourage revisiting nodes.
#   """
#   total_distance = torch.sum(output * distances)

#   # Negative log-likelihood with masking for no revisiting nodes
#   mask = torch.ones_like(output) - torch.eye(output.size(0), dtype=torch.float).to(device)
#   masked_output = output * mask
#   log_likelihood = torch.sum(torch.log(masked_output + 1e-8))  # Add epsilon for stability

#   # Combined loss with weighting factor (adjust weight as needed)
#   weight = 0.5
#   total_loss = weight * total_distance - log_likelihood

#   return total_loss


# def tsp_loss(output, distances):
#     """
#     Calculates the loss function incorporating total distance, revisit penalty, 
#     and missing node penalty for TSP.
#     """
#     device = output.device  # Ensure device compatibility

#     if not output.requires_grad:
#         raise ValueError("The output tensor does not require gradients. Ensure it is part of the computational graph.")

#     # Total distance
#     total_distance = torch.sum(output * distances)

#     # Negative log-likelihood with masking for no revisiting nodes
#     mask = torch.ones_like(output) - torch.eye(output.size(0), dtype=torch.float).to(device)
#     masked_output = output * mask
#     log_likelihood = torch.sum(torch.log(masked_output + 1e-8))  # Add epsilon for stability

#     # Revisit penalty: penalize revisiting nodes more than once
#     revisit_penalty = torch.sum(output * (output - torch.eye(output.size(0), dtype=torch.float).to(device)))

#     # Missing node penalty: penalize missing nodes
#     row_sums = torch.sum(output, dim=1)
#     col_sums = torch.sum(output, dim=0)
#     missing_penalty = torch.sum((row_sums - 1) ** 2) + torch.sum((col_sums - 1) ** 2)

#     # Combined loss with weighting factors (adjust weights as needed)
#     weight_distance = 0.5
#     weight_revisit = 1.0
#     weight_missing = 1.0

#     total_loss = (weight_distance * total_distance - log_likelihood + 
#                   weight_revisit * revisit_penalty + weight_missing * missing_penalty)

#     return total_loss

def train_model(num_epochs, tsp_instances, num_cities):
    gnn = GNNEmbeds(in_channels=2, out_channels=128).to(device)
    transformer = TransformerTSP(hidden_dim=128, num_heads=8, num_layers=6).to(device)
    model = Hybrid(gnn, transformer, num_cities).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {param_group['lr']}")
        for coordinates, distances in tsp_instances:
            # Normalize coordinates
            coordinates = normalize_coordinates(coordinates)
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
            # print(f"distances: {distances_tensor}")
            
            optimizer.zero_grad()
            output = model(data)
            # print(output)

            loss = tsp_loss(output, distances_tensor)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
            #     if param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad.norm()}")
            #     else:
            #         print(f"No gradient for {name}")
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(tsp_instances)}')

    save_model(model, save_model_path)
    print(f'Model saved to {save_model_path}')
    return model
