import torch
import numpy as np
import pickle
from model_batched import GNNEmbeds, TransformerTSP, Hybrid
from torch_geometric.data import Data, Batch
from utils import decode_tsp_tour, calculate_total_distance
from plot import plot_tsp_graph

NUM_CITIES = 10
VALIDATION_SIZE = 2000
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = 'trained_tsp_model.pth'

# Initialize and load the trained model
gnn = GNNEmbeds(in_channels=2, out_channels=128).to(device)
transformer = TransformerTSP(hidden_dim=128, num_heads=8, num_layers=6, max_seq_length=NUM_CITIES).to(device)
model = Hybrid(gnn, transformer, NUM_CITIES).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

with open('validation_set.pkl', 'rb') as f:
    validation_set = pickle.load(f)

total_model_distance = 0
total_nn_distance = 0
total_percentage_deviation = 0
model_beats_optimal = 0
model_is_optimal = 0

