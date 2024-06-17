# this is the optimal solution for small problems. not needed though

import subprocess
import numpy as np
import tempfile
import os

def write_tsplib_file(distances, filename):
    num_cities = len(distances)
    with open(filename, 'w') as f:
        f.write(f"NAME: tsp\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"DIMENSION: {num_cities}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write(f"EDGE_WEIGHT_SECTION\n")
        for row in distances:
            f.write(' '.join(map(str, row)) + '\n')
        f.write("EOF\n")

def parse_concorde_output(output):
    lines = output.strip().split('\n')
    for line in lines:
        if line.startswith("Optimal tour:"):
            tour = list(map(int, line.split(':')[1].strip().split()))
            return tour
    return None

def solve_tsp_with_concorde(distances):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tsplib_file = os.path.join(tmpdirname, "tsp_instance.tsp")
        write_tsplib_file(distances, tsplib_file)
        
        result = subprocess.run(['concorde', tsplib_file], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Concorde failed with return code {result.returncode}")
        
        tour = parse_concorde_output(result.stdout)
        return tour

# Example usage
if __name__ == "__main__":
    num_cities = 10
    coordinates = np.random.rand(num_cities, 2)
    distances = np.sqrt(((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(-1))
    optimal_tour = solve_tsp_with_concorde(distances)
    print(f"Optimal tour: {optimal_tour}")
