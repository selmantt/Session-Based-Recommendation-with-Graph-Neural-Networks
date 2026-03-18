import torch


class Recall:
    
    def __init__(self, k=20):
        self.k = k
        self.reset()
    
    def reset(self):
        #Reset metric state
        self.hits = 0
        self.total = 0
    
    def update(self, scores, targets):
        #Update metric with batch predictions
        # Get top-K indices
        _, top_k_indices = torch.topk(scores, self.k, dim=-1)
        
        # Check if target is in top-K
        targets_expanded = targets.unsqueeze(1)
        hits = (top_k_indices == targets_expanded).any(dim=1)
        
        # Update counters
        self.hits += hits.sum().item()
        self.total += targets.shape[0]
    
    def compute(self):
        if self.total == 0:
            return 0.0
        return self.hits / self.total


class MRR:
    def __init__(self, k=20):
        self.k = k
        self.reset()
    
    def reset(self):
        #Reset metric state
        self.reciprocal_ranks = 0.0
        self.total = 0
    
    def update(self, scores, targets):
        #Update metric with batch predictions
        
        # Get top-K indices
        _, top_k_indices = torch.topk(scores, self.k, dim=-1)
        
        # Find target position in top-K
        targets_expanded = targets.unsqueeze(1)
        matches = (top_k_indices == targets_expanded)
        
        # Compute rank (1-indexed)
        ranks = matches.float() * torch.arange(
            1, self.k + 1, 
            device=scores.device
        ).float()
        
        # Each example should have single rank
        ranks = ranks.sum(dim=1)
        
        # Compute reciprocal rank
        reciprocal = torch.where(
            ranks > 0,
            1.0 / ranks,
            torch.zeros_like(ranks)
        )
        
        # Update counters
        self.reciprocal_ranks += reciprocal.sum().item()
        self.total += targets.shape[0]
    
    def compute(self):
        if self.total == 0:
            return 0.0
        return self.reciprocal_ranks / self.total


class MetricTracker:
    #track metrics
    
    def __init__(self, k=20, k_values=None):
       #initialize metric tracker
        self.k = k
        self.k_values = k_values if k_values is not None else [5, 10, 20]
        
        # Create metric for each k value
        self.recalls = {kv: Recall(kv) for kv in self.k_values}
        self.mrrs = {kv: MRR(kv) for kv in self.k_values}
    
    def reset(self):
        #Reset all metrics
        for kv in self.k_values:
            self.recalls[kv].reset()
            self.mrrs[kv].reset()
    
    def update(self, scores, targets):
        for kv in self.k_values:
            self.recalls[kv].update(scores, targets)
            self.mrrs[kv].update(scores, targets)
    
    def compute(self):
        result = {}
        for kv in self.k_values:
            result[f"recall@{kv}"] = self.recalls[kv].compute()
            result[f"mrr@{kv}"] = self.mrrs[kv].compute()
        return result
