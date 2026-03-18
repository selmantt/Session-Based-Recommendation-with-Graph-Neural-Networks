#GRU4Rec: Session-based Recommendations with Recurrent Neural Networks

import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
    #GRU based session recommender
    
    def __init__(
        self,
        n_items,
        embedding_dim=100,
        hidden_dim=100,
        n_layers=1,
        dropout=0.35  
    ):
        #Initialize GRU4Rec model
        super().__init__()
        
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Item embedding
        self.embedding = nn.Embedding(
            n_items, 
            embedding_dim, 
            padding_idx=0
        )
        
        # GRU encoder
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        
        # Output dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        #Initialize model weights
        nn.init.uniform_(self.embedding.weight[1:], -0.1, 0.1)
        
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.uniform_(self.output_proj.weight, -0.1, 0.1)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, batch):
        #GRU4Rec forward pass
        sequences = batch["sequences"]
        lengths = batch["lengths"]
        
        # Embedding
        embedded = self.embedding(sequences)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False
        )
        
        # GRU forward
        _, hidden = self.gru(packed)
        
        # Last layer hidden state
        session_repr = hidden[-1]
        session_repr = self.dropout(session_repr)
        
        # Project to embedding dimension
        session_repr = self.output_proj(session_repr)
        
        # Compute scores
        item_embeddings = self.embedding.weight
        scores = torch.matmul(session_repr, item_embeddings.T)
        
        return scores
