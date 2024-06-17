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


def train_model(num_epochs, tsp_instances, num_cities):
    gnn = GNNEmbeds(in_channels=2, out_channels=128).to(device)
    transformer = TransformerTSP(hidden_dim=128, num_heads=8, num_layers=6).to(device)
    model = Hybrid(gnn, transformer, num_cities).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

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

            loss = tsp_loss(output, distances_tensor)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(tsp_instances)}')

    save_model(model, save_model_path)
    print(f'Model saved to {save_model_path}')
    return model
