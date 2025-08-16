import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class BallDetectionTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 device: str = 'auto', learning_rate: float = 1e-4, weight_decay: float = 1e-5,
                 loss_type: str = 'mse', checkpoint_dir: str = 'experiments',
                 experiment_name: str = 'ball_detection'):
        """
        Trainer for ball detection model
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use ('auto', 'cpu', 'mps', 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            loss_type: Type of loss function ('mse', 'bce', 'combined')
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        
        # Setup device
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup loss function
        self.criterion = self._setup_loss_function()
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        print(f"🚀 Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Checkpoint directory: {self.checkpoint_dir}")
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup the loss function based on loss_type"""
        if self.loss_type == 'mse':
            return nn.MSELoss()
        elif self.loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif self.loss_type == 'combined':
            return CombinedBallLoss()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets, metadata) in enumerate(self.train_loader):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (batch_idx + 1) if batch_idx > 0 else 0
                print(f"  Batch {batch_idx}/{num_batches} | Loss: {loss.item():.6f} | Time: {avg_time:.1f}s")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, epoch: int) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, metadata) in enumerate(self.val_loader):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs: int, save_every: int = 5) -> Dict[str, List[float]]:
        """Train the model for multiple epochs"""
        print(f"🎯 Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update metrics
            self.current_epoch = epoch + 1
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            # Check if best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
            
            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"📊 Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  Time: {epoch_time:.1f}s")
            print("-" * 50)
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pt')
        
        # Return training history
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
    
    def save_checkpoint(self, filename: str):
        """Save a checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        
        print(f"📂 Checkpoint loaded: {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Best Val Loss: {self.best_val_loss:.6f}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate plot
        ax2.plot(self.learning_rates, label='Learning Rate', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved: {save_path}")
        
        plt.show()


class CombinedBallLoss(nn.Module):
    """Combined loss function for ball detection"""
    
    def __init__(self, mse_weight: float = 0.7, bce_weight: float = 0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.bce_weight = bce_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse_loss(outputs, targets)
        bce_loss = self.bce_loss(outputs, targets)
        
        combined_loss = self.mse_weight * mse_loss + self.bce_weight * bce_loss
        return combined_loss


if __name__ == "__main__":
    # Test the trainer
    print("Testing BallDetectionTrainer...")
    
    # Create a dummy model and data loaders for testing
    from models.ball_detector import BallDetector
    
    model = BallDetector(n_channels=9, n_classes=1, bilinear=True)
    
    # Create dummy data loaders (this is just for testing the trainer class)
    print("✅ Trainer class created successfully!")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
