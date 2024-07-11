import numpy as np
from sklearn.utils import shuffle  # Add this import



def normalize_coordinates(coordinates):
    coordinates = np.array(coordinates)
    min_val = coordinates.min(axis=0)
    max_val = coordinates.max(axis=0)
    normalized = (coordinates - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero
    return normalized

def generate_tsp_instance(num_cities):
    coordinates = np.random.rand(num_cities, 2)
    # print(f"not normalized {coordinates}")
    # distances = np.sqrt(((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(-1))
    # print(f"not normalized distances: {distances}")
    coordinates = normalize_coordinates(coordinates)
    # print(f"normalized: {coordinates}")
    distances = np.sqrt(((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(-1))
    # print(f"normalized distances: {distances}")
    return coordinates, distances

def generate_tsp_instances(num_instances, num_cities):
    instances = []
    for _ in range(num_instances):
        coordinates, distances = generate_tsp_instance(num_cities)
        instances.append((coordinates, distances))
    return instances


def create_batches(tsp_instances, batch_size):
    batches = []
    for i in range(0, len(tsp_instances), batch_size):
        batch = tsp_instances[i:i + batch_size]
        batches.append(batch)
    return batches


# FOR VALIDATION:

# this is NN
def solve_tsp_exact(distances):
  """
  Finds an approximate optimal tour using the nearest neighbor heuristic (excluding last leg).
  """
  n = distances.shape[0]
  visited = np.zeros(n, dtype=bool)
  tour = [0]  # Start at the first city
  visited[0] = True

  for _ in range(1, n):
    min_dist_index = None
    min_dist = float('inf')
    for i in range(n):
      if not visited[i] and distances[tour[-1], i] < min_dist:
        min_dist_index = i
        min_dist = distances[tour[-1], i]
    tour.append(min_dist_index)
    visited[min_dist_index] = True

  # Calculate total distance of the tour (excluding last leg)
  total_distance = sum(distances[tour[i], tour[i + 1]] for i in range(n - 1))

  return tour, total_distance

def generate_tsp_instances_validation(num_instances, num_cities):
  """
  Generates a set of TSP instances with city coordinates, distances, and
  optimal tour information using the CBC exact solver.
  """
  instances = []
  for _ in range(num_instances):
    coordinates, distances = generate_tsp_instance(num_cities)
    optimal_tour, optimal_distance = solve_tsp_exact(distances.copy())
    instances.append((coordinates, distances, optimal_tour, optimal_distance))
  return shuffle(instances)