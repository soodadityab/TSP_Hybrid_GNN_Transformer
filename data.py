import numpy as np

def generate_tsp_instance(num_cities):
    coordinates = np.random.rand(num_cities, 2)
    distances = np.sqrt(((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(-1))
    return coordinates, distances

def generate_tsp_instances(num_instances, num_cities):
    instances = []
    for _ in range(num_instances):
        coordinates, distances = generate_tsp_instance(num_cities)
        instances.append((coordinates, distances))
    return instances
