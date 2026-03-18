import time

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .metrics import MetricTracker


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        device="cpu",
        max_grad_norm=5.0,
        checkpoint_manager=None,
        log_interval=100
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.checkpoint_manager = checkpoint_manager
        self.log_interval = log_interval
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_metrics = []
        self.best_epoch = 0
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            batch = self._move_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            scores = self.model(batch)
            
            # Compute loss
            loss = self.criterion(scores, batch["targets"])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / n_batches
                tqdm.write(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                    f"Loss = {avg_loss:.4f}"
                )
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def evaluate(self, eval_loader, k=20, desc="Eval"):
        self.model.eval()
        
        metric_tracker = MetricTracker(k=k)
        
        pbar = tqdm(eval_loader, desc=f"[{desc}]", leave=False)
        
        for batch in pbar:
            batch = self._move_to_device(batch)
            scores = self.model(batch)
            metric_tracker.update(scores, batch["targets"])
        
        return metric_tracker.compute()
    
    def train(
        self,
        train_loader,
        val_loader,
        n_epochs,
        patience=5,
        metric_key="recall@20",
        start_epoch=0,
        config=None
    ):
        print(f"\n{'='*60}")
        print(f"Training Started")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {n_epochs}")
        print(f"Patience: {patience}")
        print(f"Metric: {metric_key}")
        print(f"{'='*60}\n")
        
        best_metric = float("-inf")
        patience_counter = 0
        
        for epoch in range(start_epoch, n_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch + 1)
            self.train_losses.append(train_loss)
            
            # Validation
            val_metrics = self.evaluate(val_loader, desc="Val")
            self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
            
            epoch_time = time.time() - epoch_start
            
            # Print summary
            print(f"\nEpoch {epoch + 1}/{n_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Recall@5:  {val_metrics.get('recall@5', 0):.4f}  |  MRR@5:  {val_metrics.get('mrr@5', 0):.4f}")
            print(f"  Val Recall@10: {val_metrics.get('recall@10', 0):.4f}  |  MRR@10: {val_metrics.get('mrr@10', 0):.4f}")
            print(f"  Val Recall@20: {val_metrics['recall@20']:.4f}  |  MRR@20: {val_metrics['mrr@20']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Check improvement
            current_metric = val_metrics[metric_key]
            
            if current_metric > best_metric:
                best_metric = current_metric
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best
                if self.checkpoint_manager is not None:
                    self.checkpoint_manager.save_best(
                        self.model,
                        self.optimizer,
                        epoch + 1,
                        val_metrics,
                        metric_key,
                        self.scheduler,
                        train_losses=self.train_losses,
                        val_metrics=self.val_metrics,
                        config=config
                    )
                
                print(f"  *** New best! ***")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
            
            # Save checkpoint
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    val_metrics,
                    self.train_losses,
                    self.val_metrics,
                    self.scheduler,
                    config
                )
            
            if patience_counter >= patience:
                print(f"\nEarly stopping! No improvement for {patience} epochs.")
                break
        
        print(f"\n{'='*60}")
        print(f"Training Completed")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best {metric_key}: {best_metric:.4f}")
        print(f"{'='*60}\n")
        
        return {
            "train_losses": self.train_losses,
            "val_metrics": self.val_metrics,
            "best_epoch": self.best_epoch
        }
    
    def _move_to_device(self, batch):
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
