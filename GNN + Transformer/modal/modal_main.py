import modal
from modal import App, Volume
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy as np
from unbatched.model import GNNEmbeds, TransformerTSP, Hybrid
import torch.optim.lr_scheduler as lr_scheduler
import pickle

image = modal.Image.debian_slim().pip_install("torch", "torch-geometric", "numpy")
# Initialize the Modal app
app = modal.App("tsp-training")
volume_name = "my-model-volume"
volume = modal.Volume.from_name(volume_name, create_if_missing=True)

# Ensure the model and data are on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def save_model(model, path):
    torch.save(model.state_dict(), path)

save_model_path = '/model/trained_tsp_model'

@app.function(image=image, volumes={"/model": volume})
def normalize_coordinates(coordinates):
    coordinates = np.array(coordinates)
    min_val = coordinates.min(axis=0)
    max_val = coordinates.max(axis=0)
    normalized = (coordinates - min_val) / (max_val - min_val)
    return normalized

@app.function(image=image, gpu="any", volumes={"/model": volume})
def tsp_loss(output, distances):
    """
    Calculates the loss function incorporating total distance and negative log-likelihood
    to discourage revisiting nodes.
    """
    total_distance = torch.sum(output * distances)

    # Negative log-likelihood with masking for no revisiting nodes
    mask = torch.ones_like(output) - torch.eye(output.size(0), dtype=torch.float).to(device)
    masked_output = output * mask
    log_likelihood = torch.sum(torch.log(masked_output + 1e-8))  # Add epsilon for stability

    # Combined loss with weighting factor (adjust weight as needed)
    weight = 0.5
    total_loss = weight * total_distance - log_likelihood

    return total_loss

@app.function(image=image, gpu="any", timeout=3000, volumes={"/model": volume})
def train_model(num_epochs, tsp_instances, num_cities):
    gnn = GNNEmbeds(in_channels=2, out_channels=128).to(device)
    transformer = TransformerTSP(hidden_dim=128, num_heads=8, num_layers=6).to(device)
    model = Hybrid(gnn, transformer, num_cities).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for coordinates, distances in tsp_instances:
            # Normalize coordinates
            coordinates = normalize_coordinates.remote(coordinates)
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

            loss = tsp_loss.remote(output, distances_tensor)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(tsp_instances)}')

    save_model(model, save_model_path)
    volume.commit()
    print(f'Model saved to {save_model_path}')

# Define the main entry point for local execution
@app.local_entrypoint()
def main():
    # Generate TSP Instances
    from data import generate_tsp_instances
    from plot import plot_tsp_graph
    from utils import decode_tsp_tour, calculate_total_distance

    num_instances = 10
    num_cities = 10
    tsp_instances = generate_tsp_instances(num_instances, num_cities)

    print("Created TSP instances, starting training")

    # Train Hybrid Model
    num_epochs = 5
    train_model.remote(num_epochs, tsp_instances, num_cities)
    print("Training complete")

    device = torch.device('cpu')
    gnn = GNNEmbeds(in_channels=2, out_channels=128).to(device)
    transformer = TransformerTSP(hidden_dim=128, num_heads=8, num_layers=6).to(device)
    trained_model = Hybrid(gnn, transformer, num_cities)
    trained_model.load_state_dict(torch.load(save_model_path, map_location=device))
    print("here")
    # Plot the result
    coordinates, distances = tsp_instances[0]
    edge_index = torch.tensor(np.array(np.meshgrid(range(len(coordinates)), range(len(coordinates)))).reshape(2, -1), dtype=torch.long).to(device)
    x = torch.tensor(coordinates, dtype=torch.float).to(device)
    data = Data(x=x, edge_index=edge_index).to(device)

    # Get model output
    output = trained_model(data)

    print(output.shape)  # Should be [num_cities, num_cities]

    # Decode the TSP tour
    predicted_tour = decode_tsp_tour(output, num_cities)
    predicted_tour = predicted_tour[0]
    print(f"Predicted tour: {predicted_tour}")
    total_distance = calculate_total_distance(predicted_tour, coordinates)
    print(f"Total distance of the predicted tour: {total_distance}")

    # Plot the TSP graph with the predicted tour
    plot_tsp_graph(coordinates, distances, predicted_tour)

if __name__ == "__main__":
    main()
