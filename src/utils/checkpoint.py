#Checkpoint management

import os

import torch
import torch.nn as nn


class ColabCheckpointManager:
    def __init__(
        self,
        checkpoint_dir,
        model_name="model",
        max_checkpoints=3
    ):
       # Initialize checkpoint manager  
       
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.max_checkpoints = max_checkpoints
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Best model tracking
        self.best_metric = float("-inf")
        self.best_checkpoint_path = None
        
        # Checkpoint history 
        self.checkpoint_history = []
    
    def save(
        self,
        model,
        optimizer,
        epoch,
        metrics,
        train_losses=None,
        val_metrics=None,
        scheduler=None,
        config=None,
    ):
        # Save checkpoint
        # Create checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_metric": self.best_metric
        }
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if train_losses is not None:
            checkpoint["train_losses"] = train_losses
            
        if val_metrics is not None:
            checkpoint["val_metrics"] = val_metrics
        
        if config is not None:
            checkpoint["config"] = config
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{self.model_name}_epoch_{epoch:03d}.pt"
        )
        
        # Save
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Add to history
        self.checkpoint_history.append(checkpoint_path)
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def save_best(
        self,
        model,
        optimizer,
        epoch,
        metrics,
        metric_key="recall@20",
        scheduler=None,
        **kwargs
    ):
        
        # Save if best model
        
        current_metric = metrics.get(metric_key, float("-inf"))
        
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            
            best_path = os.path.join(
                self.checkpoint_dir,
                f"{self.model_name}_best.pt"
            )
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "best_metric": self.best_metric,
                **kwargs
            }
            
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            
            # Save
            torch.save(checkpoint, best_path)
            self.best_checkpoint_path = best_path
            
            print(f"New best model! {metric_key}={current_metric:.4f}")
            return True
        
        return False
    
    def load(
        self,
        checkpoint_path,
        model,
        optimizer=None,
        scheduler=None,
        device="cpu",
        strict=True
    ):
        
        #Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check model sizes 
        checkpoint_state = checkpoint["model_state_dict"]
        model_state = model.state_dict()
        
        # Check embedding dimension
        if "embedding.weight" in checkpoint_state and "embedding.weight" in model_state:
            checkpoint_n_items = checkpoint_state["embedding.weight"].shape[0]
            model_n_items = model_state["embedding.weight"].shape[0]
            
            if checkpoint_n_items != model_n_items:
                error_msg = (
                    f"\nModel sizes don't match!\n"
                )
                
                if strict:
                    raise RuntimeError(error_msg)
                else:
                    print(f"Warning: {error_msg}")
                    print("starting training from scratch")
                    return {
                        "epoch": 0,
                        "metrics": {},
                        "best_metric": float("-inf")
                    }
        
        # Load model state
        try:
            model.load_state_dict(checkpoint_state, strict=strict)
        except RuntimeError as e:
            if strict:
                raise
            else:
                print(f"Error loading checkpoint: {e}")
                print("Skipping checkpoint, starting training from scratch...")
                return {
                    "epoch": 0,
                    "metrics": {},
                    "best_metric": float("-inf")
                }
        
        # Load optimizer state 
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Move optimizer states to correct device
            self._move_optimizer_to_device(optimizer, device)
        
        # Load scheduler state 
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Restore best metric
        self.best_metric = checkpoint.get("best_metric", float("-inf"))
        
        return checkpoint
    
    def _move_optimizer_to_device(self, optimizer, device):
        # Move optimizer states to specified device
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    def load_best(
        self,
        model,
        optimizer=None,
        scheduler=None,
        device="cpu"
    ):
        #Load best checkpoint
        
        best_path = os.path.join(
            self.checkpoint_dir,
            f"{self.model_name}_best.pt"
        )
        
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
        
        return self.load(best_path, model, optimizer, scheduler, device)
    
    def get_latest_checkpoint(self):
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(self.model_name) and f.endswith(".pt")
            and "best" not in f
        ]
        
        if not checkpoints:
            return None
        
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].replace(".pt", "")))
        
        return os.path.join(self.checkpoint_dir, checkpoints[-1])
    
    def _cleanup_old_checkpoints(self):

        regular_checkpoints = [
            cp for cp in self.checkpoint_history
            if "best" not in cp and os.path.exists(cp)
        ]
        
        # Delete
        while len(regular_checkpoints) > self.max_checkpoints:
            old_checkpoint = regular_checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                print(f"Old checkpoint deleted: {old_checkpoint}")
            
            if old_checkpoint in self.checkpoint_history:
                self.checkpoint_history.remove(old_checkpoint)
