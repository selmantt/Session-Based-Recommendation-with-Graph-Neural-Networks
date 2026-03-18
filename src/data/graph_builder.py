#Session Graph Builder

import numpy as np
import torch


class SessionGraphBuilder:
    #Build graph from sessions
    
    @staticmethod
    def build_graph(sequence):
        #Build graph from sequence
        # Get unique items
        items = list(dict.fromkeys(sequence))
        n_nodes = len(items)
        
        # Item to node mapping
        item_to_node = {item: idx for idx, item in enumerate(items)}
        
        # Initialize adjacency matrices
        A_in = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        A_out = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        
        # Build edges from pairs
        for i in range(len(sequence) - 1):
            src = item_to_node[sequence[i]]      
            dst = item_to_node[sequence[i + 1]]  
            
            # Outgoing edge
            A_out[src, dst] += 1
            
            # Incoming edge
            A_in[dst, src] += 1
        
        # Normalize by row
        A_in = SessionGraphBuilder._normalize_adjacency(A_in)
        A_out = SessionGraphBuilder._normalize_adjacency(A_out)
        
        # Create alias
        alias = np.array([item_to_node[item] for item in sequence], dtype=np.int64)
        
        return np.array(items), A_in, A_out, alias, n_nodes
    
    @staticmethod
    def _normalize_adjacency(A):
        #Normalize adjacency matrix
        row_sum = A.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1, row_sum)
        return A / row_sum
    
    @staticmethod
    def build_batch_graphs(sequences, targets):
        #Build batch graph representation
        batch_size = len(sequences)
        
        # Build graph for each session
        graphs = [SessionGraphBuilder.build_graph(seq) for seq in sequences]
        
        # Find max dimensions
        max_nodes = max(g[4] for g in graphs)
        max_seq_len = max(len(seq) for seq in sequences)
        
        # Initialize padded tensors
        items = np.zeros((batch_size, max_nodes), dtype=np.int64)
        A_in = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
        A_out = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
        alias = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask = np.zeros((batch_size, max_nodes), dtype=np.float32)
        
        # Place each graph
        for i, (seq_items, a_in, a_out, seq_alias, n_nodes) in enumerate(graphs):
            items[i, :n_nodes] = seq_items
            A_in[i, :n_nodes, :n_nodes] = a_in
            A_out[i, :n_nodes, :n_nodes] = a_out
            alias[i, :len(seq_alias)] = seq_alias
            mask[i, :n_nodes] = 1.0
        
        # Convert to tensors
        return {
            "items": torch.LongTensor(items),
            "A_in": torch.FloatTensor(A_in),
            "A_out": torch.FloatTensor(A_out),
            "alias": torch.LongTensor(alias),
            "mask": torch.FloatTensor(mask),
            "seq_lengths": torch.LongTensor([len(seq) for seq in sequences]),
            "targets": torch.LongTensor(targets)
        }
