import torch
import numpy as np

# beam search:
# def decode_tsp_tour(softmax_output, beam_size, distance_matrix):
#     """
#     Perform beam search decoding to find the best paths for TSP based on distance.

#     Args:
#     - softmax_output (torch.Tensor): Softmaxed output from the model with shape [batch_size, num_nodes, num_cities].
#     - beam_size (int): The size of the beam.
#     - distance_matrix (torch.Tensor): A matrix containing distances between nodes with shape [num_cities, num_cities].

#     Returns:
#     - best_sequences (list): The best decoded sequences (paths) for each graph in the batch.
#     """
#     batch_size, num_nodes, num_cities = softmax_output.shape

#     best_sequences = []

#     for batch_idx in range(batch_size):
#         beam = [(0, [])]  # (score, path)

#         for node in range(num_nodes):
#             new_beam = []

#             for score, path in beam:
#                 probs = softmax_output[batch_idx, node]
#                 top_probs, top_indices = probs.topk(beam_size)

#                 for prob, next_step in zip(top_probs, top_indices):
#                     if next_step.item() in path:
#                         continue  # Skip if the next step is already in the path

#                     new_path = path + [next_step.item()]
#                     new_score = score + torch.log(prob).item()
#                     new_beam.append((new_score, new_path))

#             # Sort the new beam by scores
#             new_beam.sort(key=lambda x: x[0], reverse=True)
#             beam = new_beam[:beam_size]

#         # Evaluate paths by their total distance
#         best_distance = float('inf')
#         best_path = None
#         for _, path in beam:
#             if len(path) == num_cities:
#                 path_distance = calculate_total_distance(path, distance_matrix)
#                 if path_distance < best_distance:
#                     best_distance = path_distance
#                     best_path = path

#         return best_path

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
