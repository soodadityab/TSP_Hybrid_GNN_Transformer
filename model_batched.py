import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch_geometric.data import Data
import torch.nn as nn
import math
import torch.nn.init as init
import numpy as np


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


# should try with GAT too later

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
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # print(f"x shape: {x.shape}")
        # print(f"edge_index shape: {edge_index.shape}")
        # print(f"edge_attr shape: {edge_attr.shape}")
        x = self.convLayer1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.convLayer2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.convLayer3(x, edge_index, edge_attr)
        return x

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
        # src is batch x num_nodes x 128
        src = src + self.positional_encoding[:, :src.size(1), :]
        memory = self.encoder(src)
        tgt = tgt + self.positional_encoding[:, :tgt.size(1), :]
        output = self.decoder(tgt, memory)
        return output  # batch_size x num_nodes x 128


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

    def forward(self, data, optimal_tour):
        # Create embeddings for the nodes
        node_embeddings = self.gnn(data)
        
        # Reshape node embeddings to (batch_size, num_cities, hidden_dim)
        batch_size = data.batch.max().item() + 1
        node_embeddings = node_embeddings.view(batch_size, self.num_cities, -1)
        
        # Prepare target embeddings for the transformer
        tgt = torch.stack([node_embeddings[i][optimal_tour[i]] for i in range(batch_size)])
        
        # Ensure tgt shape is correct
        # print(tgt.shape)
        
        transformer_output = self.transformer(node_embeddings, tgt)
        logits = self.fc_out(transformer_output)  # batch_size x num_nodes x num_nodes
        output = F.softmax(logits, dim=2)  # batch_size x num_nodes x num_nodes
        # print(output)
        return output