#SR-GNN: Session-based Recommendation with Graph Neural Networks

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedGraphLayer(nn.Module):
    #Gated Graph Neural Network layer with residual
    
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.W_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden, A_in, A_out):
        #One step GGNN propagation with residual
        batch_size, n_nodes, _ = hidden.shape
        
        # Save for residual
        residual = hidden
        
        # Aggregate from neighbors
        msg_in = torch.bmm(A_in, hidden)
        msg_in = self.W_in(msg_in)
        
        msg_out = torch.bmm(A_out, hidden)
        msg_out = self.W_out(msg_out)
        
        # Combine messages
        message = msg_in + msg_out
        message = self.dropout(message)
        
        # GRU update
        hidden_flat = hidden.view(-1, self.hidden_dim)
        message_flat = message.view(-1, self.hidden_dim)
        hidden_new = self.gru(message_flat, hidden_flat)
        hidden_new = hidden_new.view(batch_size, n_nodes, self.hidden_dim)
        
        # Residual connection and layer norm
        hidden_new = self.layer_norm(hidden_new + residual)
        
        return hidden_new


class MultiHeadSessionAttention(nn.Module):
    """Multi-Head Session Attention for richer representations"""
    
    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        # Multi-head projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Learnable temperature per head
        self.temperature = nn.Parameter(torch.ones(n_heads))
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, node_hidden, last_hidden, mask):
        """Compute multi-head attention-weighted global preference"""
        batch_size, n_nodes, _ = node_hidden.shape
        
        # Query from last item, Key/Value from all nodes
        query = self.W_q(last_hidden).unsqueeze(1)  # [B, 1, D]
        key = self.W_k(node_hidden)                  # [B, N, D]
        value = self.W_v(node_hidden)                # [B, N, D]
        
        # Reshape for multi-head: [B, n_heads, seq_len, head_dim]
        query = query.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, n_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with learnable temperature
        # [B, n_heads, 1, head_dim] @ [B, n_heads, head_dim, N] = [B, n_heads, 1, N]
        scores = torch.matmul(query, key.transpose(-2, -1))
        scale = self.temperature.view(1, self.n_heads, 1, 1) * math.sqrt(self.head_dim)
        scores = scores / (scale + 1e-8)
        
        # Mask padding: expand mask for multi-head [B, 1, 1, N]
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(mask_expanded == 0, -1e9)
        
        # Softmax attention
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Weighted sum: [B, n_heads, 1, head_dim]
        context = torch.matmul(attention, value)
        
        # Reshape back: [B, D]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        context = context.squeeze(1)
        
        # Output projection
        global_pref = self.W_o(context)
        global_pref = self.layer_norm(global_pref + last_hidden)  # Residual
        
        return global_pref


# Keep old class for backward compatibility
class SessionAttention(nn.Module):
    #Session attention mechanism 
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.W_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, node_hidden, last_hidden, mask):
        #Compute attention-weighted global preference
        batch_size, n_nodes, _ = node_hidden.shape
        
        last_expanded = last_hidden.unsqueeze(1).expand(-1, n_nodes, -1)
        
        # Compute attention scores with sigmoid
        combined = self.W_1(node_hidden) + self.W_2(last_expanded)
        scores = self.v(torch.sigmoid(combined)).squeeze(-1)
        
        # Apply temperature scaling
        scores = scores / (self.temperature + 1e-8)
        
        # Mask padding
        scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax attention
        attention = F.softmax(scores, dim=-1)
        
        # Weighted sum
        global_pref = torch.bmm(
            attention.unsqueeze(1),
            node_hidden
        ).squeeze(1)
        
        # Transform global preference
        global_pref = self.W_3(global_pref)
        
        return global_pref


class PositionEncoding(nn.Module):
    #Learnable position encoding
    
    def __init__(self, hidden_dim, max_len=200):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
    
    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        return self.pos_embedding(positions)


class SRGNN(nn.Module):
    #SR-GNN: Graph Neural Network based Session Recommender
    
    def __init__(
        self,
        n_items,
        embedding_dim=100,
        hidden_dim=100,
        n_gnn_layers=1,
        n_attention_heads=4,
        dropout=0.3
    ):
        super().__init__()
        
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        
        # Item embedding
        self.embedding = nn.Embedding(
            n_items, 
            embedding_dim, 
            padding_idx=0
        )
        
        # Position encoding
        self.pos_encoding = PositionEncoding(hidden_dim)
        
        # GGNN layers
        self.gnn_layers = nn.ModuleList([
            GatedGraphLayer(hidden_dim, dropout=dropout)
            for _ in range(n_gnn_layers)
        ])
        
        # Multi-head attention for global preference
        self.attention = MultiHeadSessionAttention(
            hidden_dim, 
            n_heads=n_attention_heads, 
            dropout=dropout
        )
        
        # Layer norm before final projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Hybrid representation with gate
        self.W_hybrid = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        #Initialize model weights
        # Uniform init for embedding like original paper
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        self.embedding.weight.data[1:].uniform_(-stdv, stdv)
        
        for name, param in self.named_parameters():
            if 'embedding' in name:
                continue
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, batch):
        #SR-GNN forward pass
        items = batch["items"]
        A_in = batch["A_in"]
        A_out = batch["A_out"]
        alias = batch["alias"]
        mask = batch["mask"]
        seq_lengths = batch["seq_lengths"]
        
        batch_size = items.shape[0]
        n_nodes = items.shape[1]
        
        # Embedding
        hidden = self.embedding(items)
        
        # Add position encoding to nodes
        pos_enc = self.pos_encoding(n_nodes, items.device)
        hidden = hidden + pos_enc.unsqueeze(0)
        
        # GGNN propagation
        for gnn in self.gnn_layers:
            hidden = gnn(hidden, A_in, A_out)
        
        # Alias to original sequence
        batch_indices = torch.arange(batch_size, device=items.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, alias.shape[1])
        seq_hidden = hidden[batch_indices, alias]
        
        # Last item = local preference
        last_indices = (seq_lengths - 1).clamp(min=0)
        last_hidden = seq_hidden[
            torch.arange(batch_size, device=items.device),
            last_indices
        ]
        
        # Global preference via attention
        global_hidden = self.attention(hidden, last_hidden, mask)
        
        # Hybrid representation with gate
        hybrid = torch.cat([last_hidden, global_hidden], dim=-1)
        gate_weight = torch.sigmoid(self.gate(hybrid))
        session_repr = gate_weight * self.W_hybrid(hybrid)
        session_repr = self.final_norm(session_repr)
        session_repr = self.dropout(session_repr)
        
        # Compute scores
        item_embeddings = self.embedding.weight
        scores = torch.matmul(session_repr, item_embeddings.T)
        
        return scores
