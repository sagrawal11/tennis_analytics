#!/usr/bin/env python3
"""
Test script for the custom training framework

This script tests each component individually to ensure everything works
before running the full training pipeline.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
from models.ball_detector import BallDetector
from data.tennis_dataset import TennisBallDataset, create_data_loaders
from training.trainer import BallDetectionTrainer


def test_model():
    """Test the model architecture"""
    print("🧪 Testing model...")
    
    try:
        # Create model
        model = BallDetector(n_channels=9, n_classes=1, bilinear=True)
        
        # Test forward pass
        x = torch.randn(1, 9, 720, 1280)
        output = model(x)
        
        # Verify output shape
        expected_shape = (1, 1, 720, 1280)
        if output.shape == expected_shape:
            print(f"✅ Model test passed!")
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        else:
            print(f"❌ Model test failed!")
            print(f"  Expected: {expected_shape}")
            print(f"  Got: {output.shape}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Model test failed with error: {e}")
        return False


def test_dataset():
    """Test the dataset"""
    print("\n🧪 Testing dataset...")
    
    try:
        # Test dataset creation
        dataset = TennisBallDataset(
            csv_path='../datasets/trackNet/labels_train.csv',
            gt_dir='../datasets/trackNet/gts',
            input_height=720,
            input_width=1280
        )
        
        if len(dataset) > 0:
            # Test first sample
            input_frames, gt_heatmap, metadata = dataset[0]
            
            print(f"✅ Dataset test passed!")
            print(f"  Dataset size: {len(dataset)}")
            print(f"  Input shape: {input_frames.shape}")
            print(f"  GT shape: {gt_heatmap.shape}")
            print(f"  Input range: [{input_frames.min():.3f}, {input_frames.max():.3f}]")
            print(f"  GT range: [{gt_heatmap.min():.3f}, {gt_heatmap.max():.3f}]")
            return True
        else:
            print("❌ Dataset is empty!")
            return False
            
    except Exception as e:
        print(f"❌ Dataset test failed with error: {e}")
        return False


def test_data_loaders():
    """Test data loader creation"""
    print("\n🧪 Testing data loaders...")
    
    try:
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_csv='../datasets/trackNet/labels_train.csv',
            val_csv='../datasets/trackNet/labels_val.csv',
            batch_size=1,  # Small batch size for testing
            num_workers=0,  # No workers for testing
            input_height=720,
            input_width=1280,
            gt_dir='../datasets/trackNet/gts'
        )
        
        print(f"✅ Data loader test passed!")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        
        # Test one batch
        for inputs, targets, metadata in train_loader:
            print(f"  Batch shapes:")
            print(f"    Inputs: {inputs.shape}")
            print(f"    Targets: {targets.shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed with error: {e}")
        return False


def test_trainer():
    """Test the trainer"""
    print("\n🧪 Testing trainer...")
    
    try:
        # Create a small model and data loaders for testing
        model = BallDetector(n_channels=9, n_classes=1, bilinear=True)
        
        # Create minimal data loaders
        train_loader, val_loader = create_data_loaders(
            train_csv='../datasets/trackNet/labels_train.csv',
            val_csv='../datasets/trackNet/labels_val.csv',
            batch_size=1,
            num_workers=0,
            input_height=720,
            input_width=1280,
            gt_dir='../datasets/trackNet/gts'
        )
        
        # Create trainer
        trainer = BallDetectionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='auto',
            learning_rate=1e-4,
            weight_decay=1e-5,
            loss_type='mse',
            checkpoint_dir='test_experiments',
            experiment_name='test'
        )
        
        print(f"✅ Trainer test passed!")
        print(f"  Device: {trainer.device}")
        print(f"  Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer test failed with error: {e}")
        return False


def test_training_step():
    """Test a single training step"""
    print("\n🧪 Testing training step...")
    
    try:
        # Create model and trainer
        model = BallDetector(n_channels=9, n_classes=1, bilinear=True)
        
        train_loader, val_loader = create_data_loaders(
            train_csv='../datasets/trackNet/labels_train.csv',
            val_csv='../datasets/trackNet/labels_val.csv',
            batch_size=1,
            num_workers=0,
            input_height=720,
            input_width=1280,
            gt_dir='../datasets/trackNet/gts'
        )
        
        trainer = BallDetectionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device='auto',
            learning_rate=1e-4,
            weight_decay=1e-5,
            loss_type='mse',
            checkpoint_dir='test_experiments',
            experiment_name='test'
        )
        
        # Test one training step
        trainer.model.train()
        for inputs, targets, metadata in train_loader:
            # Move to device
            inputs = inputs.to(trainer.device)
            targets = targets.to(trainer.device)
            
            # Forward pass
            trainer.optimizer.zero_grad()
            outputs = trainer.model(inputs)
            
            # Calculate loss
            loss = trainer.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            trainer.optimizer.step()
            
            print(f"✅ Training step test passed!")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Output shape: {outputs.shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"❌ Training step test failed with error: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing Custom Training Framework")
    print("=" * 50)
    
    tests = [
        ("Model", test_model),
        ("Dataset", test_dataset),
        ("Data Loaders", test_data_loaders),
        ("Trainer", test_trainer),
        ("Training Step", test_training_step)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for training.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
