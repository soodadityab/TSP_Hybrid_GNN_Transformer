import torch
import numpy as np


def decode_tsp_tour(prob_matrix, num_cities):
    """
    Decodes the TSP tour from the model output probability matrix.

    Args:
        prob_matrix (torch.Tensor): The output probability matrix from the model.
        num_cities (int): The number of cities in the TSP instance.

    Returns:
        list: The decoded TSP tour as a list of city indices.
    """
    prob_matrix = prob_matrix.squeeze(0)

    tour = []
    visited = [False] * num_cities

    current_node = torch.argmax(prob_matrix[0]).item()
    tour.append(current_node)
    visited[current_node] = True

    for _ in range(1, num_cities):
        prob_matrix[:, current_node] = -float('inf')
        next_node = torch.argmax(prob_matrix[current_node]).item()
        while visited[next_node]:
            prob_matrix[current_node, next_node] = -float('inf')
            next_node = torch.argmax(prob_matrix[current_node]).item()
        tour.append(next_node)
        visited[next_node] = True
        current_node = next_node

    return tour

# THIS INTUITIVELY SEEM RIGHT
# def decode_tsp_tour(tensor, training=True):
#     print(tensor)
#     n = tensor.shape[1]
#     selected_indices = []
#     used_indices = set()

#     for row in tensor[0]:  # Assuming the input is a 3D tensor with shape (1, n, n)
#         if training:
#             highest_idx = torch.argmax(row).item()
#         else:
#             sorted_indices = torch.argsort(row, descending=True)
#             for idx in sorted_indices:
#                 if idx.item() not in used_indices:
#                     highest_idx = idx.item()
#                     used_indices.add(highest_idx)
#                     break
#         selected_indices.append(highest_idx)

#     return selected_indices

# def decode_tsp_tour(prob_matrix, num_cities):
#   """
#   Decodes the output probability matrix into a TSP tour ensuring no node is visited twice
#   and the nodes are within the valid range. Also calculates the negative log-likelihood of the tour.
#   """
#   tour = []
#   visited = set()
#   log_likelihood = 0.0
#   current_node = torch.argmax(prob_matrix[0]).item()
#   tour.append(current_node)
#   visited.add(current_node)

#   for i in range(num_cities - 1):
#     # Mask visited nodes by setting their probability to -inf
#     prob_matrix[:, list(visited)] = -float('inf')  
#     next_node = torch.argmax(prob_matrix[current_node]).item()
    
#     # Check if the current_node is valid
#     if next_node < 0 or next_node >= num_cities:
#       raise ValueError(f"Decoded node index {next_node} is out of bounds.")
    
#     # Update log-likelihood
#     log_likelihood += torch.log(prob_matrix[current_node, next_node]).item()
    
#     tour.append(next_node)
#     visited.add(next_node)
#     current_node = next_node

#   return tour, log_likelihood

# def calculate_total_distance(tour, coordinates):
#     total_distance = 0.0
#     print(tour)
#     num_cities = len(tour)
#     for i in range(num_cities):
#         start_city = coordinates[tour[i]]
#         end_city = coordinates[tour[(i + 1) % num_cities]]
#         distance = np.linalg.norm(np.array(start_city) - np.array(end_city))
#         # print(f"Distance from {start_city} to {end_city}: {distance}")
#         total_distance += distance
#     return total_distance

def calculate_total_distance(tour, coordinates):
    total_distance = 0.0
    num_cities = len(tour)
    for i in range(num_cities - 1):
        start_city = coordinates[tour[i]]
        end_city = coordinates[tour[i + 1]]
        distance = np.linalg.norm(np.array(start_city) - np.array(end_city))
        total_distance += distance
    return total_distance
