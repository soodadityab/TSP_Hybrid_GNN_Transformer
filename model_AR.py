import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch_geometric.data import Data
import torch.nn as nn
import math
import torch.nn.init as init

# Weight initialization function
def weights_init(m):
    if isinstance(m, NNConv):
        for param in m.parameters():
            if param.dim() > 1:
                init.xavier_uniform_(param)
            else:
                init.zeros_(param)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Transformer):
        for p in m.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

# GNN model for node embeddings
class GNNEmbeds(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNEmbeds, self).__init__()
        self.convLayer1 = NNConv(in_channels, 128, nn.Linear(1, in_channels * 128))
        self.convLayer2 = NNConv(128, 128, nn.Linear(1, 128 * 128))
        self.convLayer3 = NNConv(128, out_channels, nn.Linear(1, 128 * out_channels))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(weights_init)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.convLayer1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.convLayer2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.convLayer3(x, edge_index, edge_attr)
        return x

# Transformer model for TSP
class TransformerTSP(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, max_seq_length):
        super(TransformerTSP, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation='relu',
            batch_first=True 
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation='relu',
            batch_first=True 
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )

    def forward(self, src, tgt):
        # Apply positional encoding
        src = src + self.positional_encoding[:, :src.size(1), :]
        memory = self.encoder(src)
        tgt = tgt + self.positional_encoding[:, :tgt.size(1), :]
        output = self.decoder(tgt, memory)
        return output

    def generate(self, src, tgt_start_token, max_length):
        src = src + self.positional_encoding[:, :src.size(1), :]
        memory = self.encoder(src)
        
        batch_size = src.size(0)
        tgt = tgt_start_token
        
        for _ in range(max_length - 1):
            tgt_with_pe = tgt + self.positional_encoding[:, :tgt.size(1), :]
            output = self.decoder(tgt_with_pe, memory)
            next_token_logits = self.fc_out(output[:, -1, :])
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
        
        return tgt

# Hybrid model combining GNN and Transformer
class Hybrid(nn.Module):
    def __init__(self, gnn, transformer, num_cities):
        super(Hybrid, self).__init__()
        self.gnn = gnn
        self.transformer = transformer
        self.fc_out = nn.Linear(128, num_cities)
        self.num_cities = num_cities

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(weights_init)

    def forward(self, data, optimal_tours=None, generate=False):
        # Create embeddings for the nodes
        node_embeddings = self.gnn(data)
        
        # Reshape node embeddings to (batch_size, num_cities, hidden_dim)
        batch_size = data.batch.max().item() + 1
        node_embeddings = node_embeddings.view(batch_size, self.num_cities, -1)
        
        if generate:
            # Start the sequence with a start token (e.g., the first city)
            tgt_start_token = node_embeddings[:, :1, :]  # Assuming the first city is the start token
            output = self.transformer.generate(node_embeddings, tgt_start_token, self.num_cities)
        else:
            # Prepare target embeddings for the transformer
            tgt = torch.stack([node_embeddings[i][optimal_tours[i]] for i in range(batch_size)])
            transformer_output = self.transformer(node_embeddings, tgt)
            output = self.fc_out(transformer_output)
        
        return output