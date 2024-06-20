import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import numpy as np
from model_batched import GNNEmbeds, TransformerTSP, Hybrid
import torch.optim.lr_scheduler as lr_scheduler
from data import solve_tsp_exact, create_batches


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
#     - output (torch.Tensor): The output probabilities from the model with shape (batch_size, num_nodes, num_nodes).
#     - distances (torch.Tensor): A batch of matrices containing the distances between nodes with shape (batch_size, num_nodes, num_nodes).
    
#     Returns:
#     - total_loss (torch.Tensor): The combined loss.
#     """
#     batch_size = output.shape[0]
#     num_nodes = output.shape[1]
    
#     total_loss = 0.0
#     for i in range(batch_size):
#         # Get the path from the model output
#         _, path_indices = torch.max(output[i], dim=-1)
        
#         # Calculate the total distance for the path
#         total_distance = torch.tensor(0.0, requires_grad=True).to(device)
#         for j in range(num_nodes - 1):
#             total_distance = total_distance + distances[i, path_indices[j], path_indices[j + 1]]
        
#         # Penalize revisiting nodes or missing nodes
#         node_visits = torch.zeros(num_nodes, device=output.device)
#         for j in range(num_nodes):
#             node_visits = node_visits + torch.nn.functional.one_hot(path_indices[j], num_classes=num_nodes).float()
        
#         revisit_penalty = torch.sum((node_visits - 1).clamp(min=0))  # Penalize revisits
#         missing_penalty = torch.sum((1 - node_visits).clamp(min=0))  # Penalize missing nodes
        
#         # Combine the distance loss and the penalties
#         alpha = 1.0  # Weight for distance
#         beta = 5.0  # Weight for revisit penalty
#         gamma = 5.0  # Weight for missing node penalty
        
#         total_loss += alpha * total_distance + beta * revisit_penalty + gamma * missing_penalty
    
#     return total_loss / batch_size

# def tsp_loss(output, distances):
#     """
#     Calculates the loss function incorporating total distance and negative log-likelihood
#     to discourage revisiting nodes.
    
#     Parameters:
#     output (torch.Tensor): The model output of shape (batch_size, num_cities, num_cities)
#     distances (torch.Tensor): The distance matrix of shape (batch_size, num_cities, num_cities)
    
#     Returns:
#     torch.Tensor: The calculated loss.
#     """
#     # Calculate total distance
#     total_distance = torch.sum(output * distances, dim=(1, 2))

#     # Negative log-likelihood with masking for no revisiting nodes
#     mask = torch.ones_like(output) - torch.eye(output.size(1), dtype=torch.float).to(output.device).unsqueeze(0)
#     masked_output = output * mask
#     log_likelihood = torch.sum(torch.log(masked_output + 1e-8), dim=(1, 2))  # Add epsilon for stability

#     # Combined loss with weighting factor (adjust weight as needed)
#     weight = 0.5
#     total_loss = weight * total_distance - log_likelihood
#     mean = torch.mean(total_loss)
#     print(mean.type)
#     return mean


def tsp_loss(output, distances, alpha=1.0, beta=2.0):
    """
    Calculates the TSP loss, minimizing distance traveled while penalizing revisited/omitted nodes.

    Args:
        output (torch.Tensor): The model output of shape (batch_size, num_cities, num_cities),
                               representing probabilities for each edge.
        distances (torch.Tensor): The distance matrix of shape (batch_size, num_cities, num_cities).
        alpha (float, optional): Weight for the distance term. Defaults to 1.0.
        beta (float, optional): Weight for the violation penalty term. Defaults to 1.0.

    Returns:
        torch.Tensor: The calculated loss.
    """
    batch_size, num_cities, _ = output.shape

    # 1. Distance Term: Minimize total tour length
    distance_term = torch.sum(output * distances, dim=(1, 2))

    # 2. Violation Penalty Term: Penalize revisited/omitted nodes
    # 2.1. Revisited Nodes: Ensure each node is visited exactly once
    visited = torch.matmul(output, torch.eye(num_cities, device=output.device))  # (batch_size, num_cities, num_cities)
    visited_penalty = torch.sum(torch.abs(visited - torch.eye(num_cities, device=output.device)), dim=(1, 2))

    # 2.2. Omitted Nodes: Penalize missing connections in the tour
    connection_penalty = torch.sum(torch.abs(torch.sum(output, dim=2) - 1), dim=1) + \
                         torch.sum(torch.abs(torch.sum(output, dim=1) - 1), dim=1)

    # Combine terms and return loss
    loss = alpha * distance_term + beta * (visited_penalty + connection_penalty)
    return loss.mean()


# def calculate_path_distance(path, distances):
#     batch_size, num_cities = path.shape
#     total_distance = torch.zeros(batch_size, device=path.device)
    
#     for i in range(batch_size):
#         for j in range(num_cities - 1):
#             total_distance[i] += distances[i, path[i, j], path[i, j + 1]]
#         # Add distance from the last to the first city to complete the tour
#         total_distance[i] += distances[i, path[i, -1], path[i, 0]]
    
#     return total_distance


# def tsp_loss(output, distances, alpha=1.0, beta=2.0, gamma=5.0):
#     """
#     Calculates the TSP loss, minimizing distance traveled while penalizing revisited/omitted nodes.

#     Args:
#         output (torch.Tensor): The model output of shape (batch_size, num_cities, num_cities),
#                                representing probabilities for each edge.
#         distances (torch.Tensor): The distance matrix of shape (batch_size, num_cities, num_cities).
#         alpha (float, optional): Weight for the distance term. Defaults to 1.0.
#         beta (float, optional): Weight for the revisit penalty term. Defaults to 2.0.
#         gamma (float, optional): Weight for the omission penalty term. Defaults to 5.0.

#     Returns:
#         torch.Tensor: The calculated loss.
#     """
#     # print(f"output: {output.shape}")
#     # print(f"output: {output}")
#     # print(f"dist: {distances.shape}")
#     # print(f"dist: {distances}")
#     batch_size, num_cities, _ = output.shape

#     _, path = torch.max(output, dim=2)
#     # print(f"Decoded Path: {path}")

#     # 1. Distance Term: Minimize total tour length based on decoded path
#     path_distance = calculate_path_distance(path, distances)
#     # print(f"Path Distance: {path_distance}")

#     # # 1. Distance Term: Minimize total tour length
#     # distance_term = torch.sum(output * distances, dim=(1, 2))
#     # # print(f"distance_term: {distance_term}")

#     # Use softmax probabilities directly for node visits
#     node_visits = torch.sum(output, dim=1)

#     # 2. Revisited Nodes Penalty: Ensure each node is visited exactly once
#     revisit_penalty = torch.sum((node_visits - 1).clamp(min=0))

#     # 3. Omitted Nodes Penalty: Penalize missing connections in the tour
#     omission_penalty = torch.sum((1 - node_visits).clamp(min=0))

#     # Combine terms and return loss
#     # loss = alpha * distance_term + beta * revisit_penalty + gamma * omission_penalty
#     loss = alpha * path_distance + beta * revisit_penalty + gamma * omission_penalty

#     return loss.mean()



def train_model(num_epochs, tsp_instances, num_cities, batch_size):
    gnn = GNNEmbeds(in_channels=2, out_channels=128).to(device)
    transformer = TransformerTSP(hidden_dim=128, num_heads=8, num_layers=6).to(device)
    model = Hybrid(gnn, transformer, num_cities).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Create batches
    batches = create_batches(tsp_instances, batch_size)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {param_group['lr']}")
        
        for batch in batches:
            optimizer.zero_grad()
            batch_loss = 0
            data_list = []
            for coordinates, distances in batch:
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
                data = Data(x=x, edge_index=edge_index, distances=distances_tensor).to(device)
                data_list.append(data)
            
            batch_data = Batch.from_data_list(data_list).to(device)
            
            # Debugging: Check shapes and values of inputs
            # print(f"Node features shape: {batch_data.x.shape}")
            # print(f"Edge index shape: {batch_data.edge_index.shape}")
            
            output = model(batch_data)
            
            # Debugging: Print the output
            # print(f"Model output shape: {output.shape}")
            # print(f"Model output: {output}")
            
            # Reshape distances to match output shape for loss computation
            distances_batch = torch.stack([data.distances for data in data_list])
            loss = tsp_loss(output, distances_batch)

            # Print the loss value
                
            batch_loss += loss
            
            batch_loss /= len(batch)  # Average the batch loss
            batch_loss.backward()

            # Check if parameters have gradients
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
            #     if param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad.norm()}")
            #     else:
            #         print(f"No gradient for {name}")
                    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_loss += batch_loss.item()
        
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(batches)}')

    save_model(model, save_model_path)
    print(f'Model saved to {save_model_path}')
    return model


# def train_model(num_epochs, tsp_instances, num_cities, batch_size):
#     gnn = GNNEmbeds(in_channels=2, out_channels=128).to(device)
#     transformer = TransformerTSP(hidden_dim=128, num_heads=8, num_layers=6).to(device)
#     model = Hybrid(gnn, transformer, num_cities).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=0.00001)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
#     # Create batches
#     batches = create_batches(tsp_instances, batch_size)
    
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         for param_group in optimizer.param_groups:
#             print(f"Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {param_group['lr']}")
        
#         for batch in batches:
#             optimizer.zero_grad()
#             batch_loss = 0
#             data_list = []
#             for coordinates, distances in batch:
#                 # Normalize coordinates
#                 coordinates = normalize_coordinates(coordinates)
#                 # Create edge index for a complete graph without self-loops
#                 edge_index = np.array(np.meshgrid(range(num_cities), range(num_cities))).reshape(2, -1)
#                 # Filter out self-loops
#                 mask = edge_index[0] != edge_index[1]
#                 edge_index = edge_index[:, mask]
#                 # Convert to PyTorch tensor and move to the appropriate device
#                 edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

#                 x = torch.tensor(coordinates, dtype=torch.float).to(device)
#                 distances_tensor = torch.tensor(distances, dtype=torch.float).to(device)
#                 data = Data(x=x, edge_index=edge_index, distances=distances_tensor).to(device)
#                 data_list.append(data)
            
#             batch_data = Batch.from_data_list(data_list).to(device)
#             output = model(batch_data)
            
#             # Reshape distances to match output shape for loss computation
#             distances_batch = torch.stack([data.distances for data in data_list])
#             loss = tsp_loss(output, distances_batch)
                
#             batch_loss += loss
            
#             batch_loss /= len(batch)  # Average the batch loss
#             batch_loss.backward()

#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
#             optimizer.step()
#             total_loss += batch_loss.item()
        
#         scheduler.step()
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(batches)}')

#     save_model(model, save_model_path)
#     print(f'Model saved to {save_model_path}')
#     return model