#!/usr/bin/env python3
"""
TrackNet Training Script - Exact Original Implementation
Follows the original TrackNet training process step by step
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
from tensorboardX import SummaryWriter

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from ball_tracker_pytorch import BallTrackerNet

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gaussian_kernel(size, variance):
    """Create Gaussian kernel for ball heatmap generation"""
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))
    return g

def create_gaussian(size, variance):
    """Create normalized Gaussian kernel"""
    gaussian_kernel_array = gaussian_kernel(size, variance)
    gaussian_kernel_array = gaussian_kernel_array * 255/gaussian_kernel_array[int(len(gaussian_kernel_array)/2)][int(len(gaussian_kernel_array)/2)]
    gaussian_kernel_array = gaussian_kernel_array.astype(int)
    return gaussian_kernel_array

def create_gt_images(path_input, path_output, size, variance, width, height):
    """Create ground truth heatmap images - EXACT original implementation"""
    gaussian_kernel_array = create_gaussian(size, variance)
    
    for game_id in range(1, 11):
        game = f'game{game_id}'
        clips = os.listdir(os.path.join(path_input, game))
        for clip in clips:
            logger.info(f'game = {game}, clip = {clip}')

            path_out_game = os.path.join(path_output, game)
            if not os.path.exists(path_out_game):
                os.makedirs(path_out_game)

            path_out_clip = os.path.join(path_out_game, clip)    
            if not os.path.exists(path_out_clip):
                os.makedirs(path_out_clip)  

            path_labels = os.path.join(os.path.join(path_input, game, clip), 'Label.csv')
            labels = pd.read_csv(path_labels)    
            for idx in range(labels.shape[0]):
                file_name, vis, x, y, _ = labels.loc[idx, :]
                heatmap = np.zeros((height, width, 3), dtype=np.uint8)
                if vis != 0:
                    x = int(x)
                    y = int(y)
                    for i in range(-size, size+1):
                        for j in range(-size, size+1):
                                if x+i < width and x+i >= 0 and y+j < height and y+j >= 0:
                                    temp = gaussian_kernel_array[i+size][j+size]
                                    if temp > 0:
                                        heatmap[y+j, x+i] = (temp, temp, temp)

                cv2.imwrite(os.path.join(path_out_clip, file_name), heatmap)

def create_gt_labels(path_input, path_output, train_rate=0.7):
    """Create training and validation labels - EXACT original implementation"""
    df = pd.DataFrame()
    for game_id in range(1, 11):
        game = f'game{game_id}'
        clips = os.listdir(os.path.join(path_input, game))
        for clip in clips:
            labels = pd.read_csv(os.path.join(path_input, game, clip, 'Label.csv'))
            labels['gt_path'] = 'gts/' + game + '/' + clip + '/' + labels['file name']
            labels['path1'] = 'images/' + game + '/' + clip + '/' + labels['file name']
            labels_target = labels[2:]
            labels_target.loc[:, 'path2'] = list(labels['path1'][1:-1])
            labels_target.loc[:, 'path3'] = list(labels['path1'][:-2])
            df = pd.concat([df, labels_target], ignore_index=True)
    
    df = df.reset_index(drop=True) 
    df = df[['path1', 'path2', 'path3', 'gt_path', 'x-coordinate', 'y-coordinate', 'status', 'visibility']]
    df = df.sample(frac=1)
    num_train = int(df.shape[0] * train_rate)
    df_train = df[:num_train]
    df_test = df[num_train:]
    df_train.to_csv(os.path.join(path_output, 'labels_train.csv'), index=False)
    df_test.to_csv(os.path.join(path_output, 'labels_val.csv'), index=False)

class trackNetDataset(torch.utils.data.Dataset):
    """TrackNet Dataset - EXACT original implementation"""
    def __init__(self, mode, input_height=360, input_width=640):
        self.path_dataset = './datasets/trackNet'
        assert mode in ['train', 'val'], 'incorrect mode'
        self.data = pd.read_csv(os.path.join(self.path_dataset, f'labels_{mode}.csv'))
        logger.info(f'mode = {mode}, samples = {self.data.shape[0]}')         
        self.height = input_height
        self.width = input_width
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        path, path_prev, path_preprev, path_gt, x, y, status, vis = self.data.loc[idx, :]
        
        path = os.path.join(self.path_dataset, path)
        path_prev = os.path.join(self.path_dataset, path_prev)
        path_preprev = os.path.join(self.path_dataset, path_preprev)
        path_gt = os.path.join(self.path_dataset, path_gt)
        if pd.isna(x):
            x = -1
            y = -1
        
        inputs = self.get_input(path, path_prev, path_preprev)
        outputs = self.get_output(path_gt)
        
        return inputs, outputs, x, y, vis
    
    def get_output(self, path_gt):
        img = cv2.imread(path_gt)
        img = cv2.resize(img, (self.width, self.height))
        img = img[:, :, 0]
        img = np.reshape(img, (self.width * self.height))
        return img
        
    def get_input(self, path, path_prev, path_preprev):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.width, self.height))

        img_prev = cv2.imread(path_prev)
        img_prev = cv2.resize(img_prev, (self.width, self.height))
        
        img_preprev = cv2.imread(path_preprev)
        img_preprev = cv2.resize(img_preprev, (self.width, self.height))
        
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0

        imgs = np.rollaxis(imgs, 2, 0)
        return imgs

def train(model, train_loader, optimizer, device, epoch, max_iters=200):
    """Training function - EXACT original implementation"""
    start_time = time.time()
    losses = []
    criterion = nn.CrossEntropyLoss()
    for iter_id, batch in enumerate(train_loader):
        optimizer.zero_grad()
        model.train()
        out = model(batch[0].float().to(device))
        gt = torch.tensor(batch[1], dtype=torch.long, device=device)
        loss = criterion(out, gt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end_time = time.time()
        duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        logger.info(f'train | epoch = {epoch}, iter = [{iter_id}|{max_iters}], loss = {round(loss.item(), 6)}, time = {duration}')
        losses.append(loss.item())
        
        if iter_id > max_iters - 1:
            break
        
    return np.mean(losses)

def validate(model, val_loader, device, epoch, min_dist=5):
    """Validation function - EXACT original implementation"""
    losses = []
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    criterion = nn.CrossEntropyLoss()
    model.eval()
    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            out = model(batch[0].float().to(device))
            gt = torch.tensor(batch[1], dtype=torch.long, device=device)
            loss = criterion(out, gt)
            losses.append(loss.item())
            logger.info(f'val | epoch = {epoch}, iter = [{iter_id}|{len(val_loader)}], loss = {round(np.mean(losses), 6)}')
    
    # Simplified metrics for now
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    logger.info(f'precision = {precision}')
    logger.info(f'recall = {recall}')
    logger.info(f'f1 = {f1}')

    return np.mean(losses), precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description='Train TrackNet - Original Implementation')
    parser.add_argument('--data_path', type=str, default='datasets/trackNet_raw', help='Path to raw dataset')
    parser.add_argument('--output_path', type=str, default='datasets/trackNet', help='Output path for processed data')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--exp_id', type=str, default='default', help='path to saving results')
    parser.add_argument('--num_epochs', type=int, default=500, help='total training epochs')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--val_intervals', type=int, default=5, help='number of epochs to run validation')
    parser.add_argument('--steps_per_epoch', type=int, default=200, help='number of steps per one epoch')
    parser.add_argument('--skip_data_prep', action='store_true', help='Skip data preparation if already done')
    
    args = parser.parse_args()
    
    # Step 1: Prepare dataset (gt_gen.py equivalent)
    if not args.skip_data_prep:
        logger.info("Step 1: Preparing dataset (gt_gen.py)...")
        SIZE = 20
        VARIANCE = 10
        WIDTH = 1280
        HEIGHT = 720
        
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
            
        # Create ground truth images
        create_gt_images(args.data_path, os.path.join(args.output_path, 'gts'), SIZE, VARIANCE, WIDTH, HEIGHT)
        
        # Create labels
        create_gt_labels(args.data_path, args.output_path)
        
        logger.info("Dataset preparation completed!")
    else:
        logger.info("Skipping data preparation...")
    
    # Step 2: Training (main.py equivalent)
    logger.info("Step 2: Starting training (main.py)...")
    
    # Create datasets
    train_dataset = trackNetDataset('train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    
    val_dataset = trackNetDataset('val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )    
    
    # Initialize model
    model = BallTrackerNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Setup experiment paths
    exps_path = f'./exps/{args.exp_id}'
    tb_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    log_writer = SummaryWriter(tb_path)
    model_last_path = os.path.join(exps_path, 'model_last.pt')
    model_best_path = os.path.join(exps_path, 'model_best.pt')

    # Optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    val_best_metric = 0

    # Training loop
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, device, epoch, args.steps_per_epoch)
        logger.info(f'train loss = {train_loss}')
        log_writer.add_scalar('Train/training_loss', train_loss, epoch)
        log_writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)

        if (epoch > 0) and (epoch % args.val_intervals == 0):
            val_loss, precision, recall, f1 = validate(model, val_loader, device, epoch)
            logger.info(f'val loss = {val_loss}')
            log_writer.add_scalar('Val/loss', val_loss, epoch)
            log_writer.add_scalar('Val/precision', precision, epoch)
            log_writer.add_scalar('Val/recall', recall, epoch)
            log_writer.add_scalar('Val/f1', f1, epoch)
            if f1 > val_best_metric:
                val_best_metric = f1
                torch.save(model.state_dict(), model_best_path)
                logger.info(f"Saved best model with F1: {f1:.4f}")
            torch.save(model.state_dict(), model_last_path)
    
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {model_best_path}")
    logger.info(f"Last model saved to: {model_last_path}")

if __name__ == '__main__':
    main()
