#!/usr/bin/env python3
"""
Tennis Ball Detection Training Script

This script trains a U-Net model for tennis ball detection using the TrackNet dataset.
The model processes 3 consecutive frames (9 channels) and outputs a heatmap for ball positions.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from models.ball_detector import BallDetector
from data.tennis_dataset import create_data_loaders
from training.trainer import BallDetectionTrainer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train tennis ball detection model')
    
    # Data arguments
    parser.add_argument('--train-csv', type=str, default='../datasets/trackNet/labels_train.csv',
                       help='Path to training CSV file')
    parser.add_argument('--val-csv', type=str, default='../datasets/trackNet/labels_val.csv',
                       help='Path to validation CSV file')
    parser.add_argument('--images-dir', type=str, default='../datasets/trackNet/images',
                       help='Directory containing input images')
    parser.add_argument('--gt-dir', type=str, default='../datasets/trackNet/gts',
                       help='Directory containing ground truth heatmaps')
    
    # Model arguments
    parser.add_argument('--input-height', type=int, default=720,
                       help='Input image height')
    parser.add_argument('--input-width', type=int, default=1280,
                       help='Input image width')
    parser.add_argument('--n-channels', type=int, default=9,
                       help='Number of input channels (3 frames * 3 RGB channels)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for training (optimized for 720x1280)')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--loss-type', type=str, default='mse',
                       choices=['mse', 'bce', 'combined'],
                       help='Loss function type')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of worker processes (reduced for memory)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'mps', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--checkpoint-dir', type=str, default='experiments',
                       help='Directory to save checkpoints')
    parser.add_argument('--experiment-name', type=str, default='ball_detection_720p',
                       help='Name of the experiment')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_arguments()
    
    print("🎾 Tennis Ball Detection Training")
    print("=" * 50)
    print(f"Training CSV: {args.train_csv}")
    print(f"Validation CSV: {args.val_csv}")
    print(f"Images directory: {args.images_dir}")
    print(f"GT directory: {args.gt_dir}")
    print(f"Input resolution: {args.input_width}x{args.input_height}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Loss type: {args.loss_type}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    # Validate paths
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")
    if not os.path.exists(args.val_csv):
        raise FileNotFoundError(f"Validation CSV not found: {args.val_csv}")
    if not os.path.exists(args.images_dir):
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    if not os.path.exists(args.gt_dir):
        raise FileNotFoundError(f"GT directory not found: {args.gt_dir}")
    
    # Create data loaders
    print("\n📊 Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_height=args.input_height,
        input_width=args.input_width,
        images_dir=args.images_dir,
        gt_dir=args.gt_dir
    )
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = BallDetector(
        n_channels=args.n_channels,
        n_classes=1,
        bilinear=True
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    print("\n🚀 Initializing trainer...")
    trainer = BallDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print(f"\n🎯 Starting training for {args.num_epochs} epochs...")
    training_history = trainer.train(
        num_epochs=args.num_epochs,
        save_every=5
    )
    
    # Plot training history
    print("\n📊 Plotting training history...")
    plot_path = Path(args.checkpoint_dir) / args.experiment_name / "training_history.png"
    trainer.plot_training_history(save_path=str(plot_path))
    
    # Save training history
    history_path = Path(args.checkpoint_dir) / args.experiment_name / "training_history.txt"
    with open(history_path, 'w') as f:
        f.write("Training History\n")
        f.write("=" * 20 + "\n")
        f.write(f"Best validation loss: {trainer.best_val_loss:.6f}\n")
        f.write(f"Total epochs: {trainer.current_epoch}\n")
        f.write(f"Final learning rate: {trainer.learning_rates[-1]:.2e}\n")
        f.write("\nEpoch-by-epoch results:\n")
        for i, (train_loss, val_loss, lr) in enumerate(zip(
            trainer.train_losses, trainer.val_losses, trainer.learning_rates
        )):
            f.write(f"Epoch {i+1}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={lr:.2e}\n")
    
    print(f"\n💾 Training history saved: {history_path}")
    print(f"🎉 Training complete! Best validation loss: {trainer.best_val_loss:.6f}")


if __name__ == "__main__":
    main()
