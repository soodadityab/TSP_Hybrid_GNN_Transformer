import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Transformer
import torch.nn as nn
import math
import torch.nn.init as init


def weights_init(m):
    if isinstance(m, GCNConv):
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

class GNNEmbeds(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNEmbeds, self).__init__()
        self.convLayer1 = GCNConv(in_channels, 128)
        self.convLayer2 = GCNConv(128, 128)
        self.convLayer3 = GCNConv(128, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(weights_init)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.convLayer1(x, edge_index)
        x = F.relu(x)
        x = self.convLayer2(x, edge_index)
        x = F.relu(x)
        x = self.convLayer3(x, edge_index)
        x = F.relu(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# class TransformerTSP(nn.Module):
#     def __init__(self, hidden_dim, num_heads, num_layers):
#         super(TransformerTSP, self).__init__()
#         self.positional_encoding = PositionalEncoding(hidden_dim)
#         self.transformer = Transformer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=num_layers,
#             batch_first=True  # this impacts the way src is inputted
#         )

#     def forward(self, src):
#         src = self.positional_encoding(src)
#         tgt = src
#         output = self.transformer(src, tgt)
#         return output

class TransformerTSP(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super(TransformerTSP, self).__init__()
        self.positional_encoding = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
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
            dropout=0.1,
            activation='relu',
            batch_first=True 
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )

    def forward(self, src, tgt):
        src = self.positional_encoding(src)
        memory = self.encoder(src)
        tgt = self.positional_encoding(tgt)
        output = self.decoder(tgt, memory)
        return output

class Hybrid(nn.Module):
    def __init__(self, gnn, transformer, num_cities):
        super(Hybrid, self).__init__()
        self.gnn = gnn
        self.transformer = transformer
        self.fc_out = nn.Linear(128, num_cities)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(weights_init)

    def forward(self, data):
        node_embeddings = self.gnn(data)
        node_embeddings = node_embeddings.unsqueeze(0)  # 1 x cities x 128 (batch_first)
        
        # Initialize the target sequence (tgt) with zeros
        tgt = torch.zeros_like(node_embeddings)
        
        transformer_output = self.transformer(node_embeddings, tgt)
        transformer_output = transformer_output.squeeze(0)  # cities x 128
        logits = self.fc_out(transformer_output)  # cities x cities
        output = F.softmax(logits, dim=1)  # cities x cities
        return output




# class Hybrid(nn.Module):
#     def __init__(self, gnn, transformer, num_cities):
#         super(Hybrid, self).__init__()
#         self.gnn = gnn
#         self.transformer = transformer
#         self.fc_out = nn.Linear(128, num_cities)
#         self.initialize_weights()

#     def initialize_weights(self):
#         self.apply(weights_init)

#     def forward(self, data):
#         node_embeddings = self.gnn(data)
#         node_embeddings = node_embeddings.unsqueeze(0)  # 1 x cities x 128
#         transformer_output = self.transformer(node_embeddings)
#         transformer_output = transformer_output.squeeze(0)  # cities x 128
#         logits = self.fc_out(transformer_output)  # cities x cities
#         output = F.softmax(logits, dim=1)  # cities x cities
#         return output