import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import numpy as np
from model_AR import GNNEmbeds, TransformerTSP, Hybrid
import torch.optim.lr_scheduler as lr_scheduler
from data import create_batches

# Ensure the model and data are on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def save_model(model, path):
    torch.save(model.state_dict(), path)

save_model_path = 'trained_tsp_model.pth'

def tsp_loss(pred_probs, optimal_tours, distances, alpha=1.0, beta=1.0):
    """
    Calculates the TSP loss, minimizing distance traveled while penalizing incorrect node order.
    
    Args:
        pred_probs (torch.Tensor): The model output of shape (batch_size, num_cities, num_cities),
                                   representing probabilities for each edge.
        optimal_tours (List[torch.Tensor]): List of optimal tours for each instance in the batch.
        distances (torch.Tensor): The distance matrix of shape (batch_size, num_cities, num_cities).
        alpha (float): Weight for the distance term.
        beta (float): Weight for the permutation penalty term.
    
    Returns:
        torch.Tensor: The calculated loss.
    """
    batch_size, num_cities, _ = pred_probs.shape
    
    # Distance Term: Minimize total tour length
    distance_term = torch.sum(pred_probs * distances, dim=(1, 2))
    
    # Permutation Penalty Term: Ensure the predicted tour matches the optimal tour
    permutation_penalty = 0
    for i in range(batch_size):
        optimal_tour = optimal_tours[i]
        optimal_tour_mask = torch.zeros(num_cities, num_cities, device=pred_probs.device)
        for j in range(num_cities - 1):
            optimal_tour_mask[optimal_tour[j], optimal_tour[j + 1]] = 1
        optimal_tour_mask[optimal_tour[-1], optimal_tour[0]] = 1  # Closing the tour
        permutation_penalty += F.binary_cross_entropy(pred_probs[i], optimal_tour_mask)
    
    permutation_penalty /= batch_size
    
    loss = alpha * distance_term + beta * permutation_penalty
    return loss.mean()

# def tsp_loss(output, distances, alpha=2.0, beta=3.0):
#     """
#     Calculates the TSP loss, minimizing distance traveled while penalizing revisited/omitted nodes.

#     Args:
#         output (torch.Tensor): The model output of shape (batch_size, num_cities, num_cities),
#                                representing probabilities for each edge.
#         distances (torch.Tensor): The distance matrix of shape (batch_size, num_cities, num_cities).
#         alpha (float, optional): Weight for the distance term. Defaults to 1.0.
#         beta (float, optional): Weight for the violation penalty term. Defaults to 1.0.

#     Returns:
#         torch.Tensor: The calculated loss.
#     """
#     # print(f"output shape: {output.shape}")
#     # print(f"distances shape: {distances.shape}")
    
#     batch_size, num_cities, _ = output.shape
#     # 1. Distance Term: Minimize total tour length
    
#     distance_term = torch.sum(output * distances, dim=(1, 2))

#     # 2. Violation Penalty Term: Penalize revisited/omitted nodes
#     # 2.1. Revisited Nodes: Ensure each node is visited exactly once
#     visited = torch.matmul(output, torch.eye(num_cities, device=output.device))  # (batch_size, num_cities, num_cities)
#     visited_penalty = torch.sum(torch.abs(visited - torch.eye(num_cities, device=output.device)), dim=(1, 2))

#     # 2.2. Omitted Nodes: Penalize missing connections in the tour
#     connection_penalty = torch.sum(torch.abs(torch.sum(output, dim=2) - 1), dim=1) + \
#                          torch.sum(torch.abs(torch.sum(output, dim=1) - 1), dim=1)

#     # Combine terms and return loss
#     # print(f"distance term: {distance_term}")
#     # print(f"visit penalty: {visited_penalty}")
#     # print(f"connection_penalty: {connection_penalty}")
#     loss = alpha * distance_term + beta * (visited_penalty + connection_penalty)
#     return loss.mean()


def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f'Gradient for {name}: {param.grad.norm().item()}')
        else:
            print(f'No gradient for {name}')


def train_model(num_epochs, tsp_instances, num_cities, batch_size):
    gnn = GNNEmbeds(in_channels=2, out_channels=128).to(device)
    transformer = TransformerTSP(hidden_dim=128, num_heads=8, num_layers=6, max_seq_length=num_cities).to(device)
    model = Hybrid(gnn, transformer, num_cities).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    batches = create_batches(tsp_instances=tsp_instances, batch_size=batch_size)
    
    # print(batches)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {param_group['lr']}")
        
        for batch in batches:
            optimizer.zero_grad()
            batch_loss = 0
            data_list = []
            optimal_tours = []
            distancesList = []
            for coordinates, distances, optimal_tour, optimal_distance in batch:
                edge_index = np.array(np.meshgrid(range(num_cities), range(num_cities))).reshape(2, -1)
                mask = edge_index[0] != edge_index[1]
                edge_index = edge_index[:, mask]
                edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

                x = torch.tensor(coordinates, dtype=torch.float).to(device)
                # Distances as edge attributes
                # Flatten the upper triangle of the distances matrix, excluding the diagonal
                row, col = edge_index
                edge_attr = distances[row, col].reshape(-1)  # Ensure edge_attr has shape [num_edges]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device).unsqueeze(-1)  # Reshape to [num_edges, 1]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)
                data_list.append(data)
                print(optimal_tour)
                optimal_tours.append(optimal_tour)
                distancesList.append(coordinates)
            
            batch_data = Batch.from_data_list(data_list).to(device)
            # print(f"data batch is {batch_data}")
            output = model(batch_data, optimal_tours, generate=False)

            # Convert edge attributes back to distance matrix shape
            distances_batch = []
            for data in data_list:
                num_edges = data.edge_index.shape[1]
                distances_matrix = torch.zeros(num_cities, num_cities, device=device)
                distances_matrix[data.edge_index[0], data.edge_index[1]] = data.edge_attr.view(-1)
                distances_batch.append(distances_matrix)
            distances_batch = torch.stack(distances_batch)
            # print(f"remade: {distances_batch}")
            loss = tsp_loss(output, optimal_tours, distances_batch)

            batch_loss += loss
            batch_loss /= len(batch)
            batch_loss.backward()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
            #     if param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad.norm()}")
            #     else:
            #         print(f"No gradient for {name}")

            # print_gradients(model)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f'NaN or Inf found in gradients of {name}')
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item()
        
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(batches)}')

    save_model(model, save_model_path)
    print(f'Model saved to {save_model_path}')
    return model