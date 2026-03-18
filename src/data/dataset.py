#Dataset Module

import torch
from torch.utils.data import Dataset

from .graph_builder import SessionGraphBuilder


class SessionDataset(Dataset):
    #SR-GNN session dataset
    
    def __init__(self, sessions):
        #Initialize dataset
        self.sessions = sessions
    
    def __len__(self):
        #Return dataset size
        return len(self.sessions)
    
    def __getitem__(self, idx):
        #Return sample at index
        return self.sessions[idx]


def collate_fn(batch):
    #SR-GNN collate function
    # Separate batch
    sequences = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Build graph batch
    return SessionGraphBuilder.build_batch_graphs(sequences, targets)


class SequenceDataset(Dataset):
    #GRU4Rec sequence dataset
    
    def __init__(self, sessions):
        #Initialize dataset
        self.sessions = sessions
    
    def __len__(self):
        #Return dataset size
        return len(self.sessions)
    
    def __getitem__(self, idx):
        #Return sample at index
        return self.sessions[idx]


def sequence_collate_fn(batch):
    #GRU4Rec collate function
    # Separate batch
    sequences = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Find max length
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    
    # Initialize padded tensors
    padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Place each sequence
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded[i, :seq_len] = torch.LongTensor(seq)
        mask[i, :seq_len] = 1.0
        lengths[i] = seq_len
    
    return {
        "sequences": padded,
        "mask": mask,
        "lengths": lengths,
        "targets": torch.LongTensor(targets)
    }
