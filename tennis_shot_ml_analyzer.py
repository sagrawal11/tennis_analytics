#!/usr/bin/env python3
"""
Tennis Shot ML Analyzer
Analyzes extracted features to learn shot classification patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TennisShotMLAnalyzer:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.features = None
        self.labels = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the data"""
        logger.info(f"Loading data from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.df)} data points with {len(self.df.columns)} features")
        
        # Show basic info
        logger.info(f"Data shape: {self.df.shape}")
        logger.info(f"Missing values: {self.df.isnull().sum().sum()}")
        
    def create_labels_from_manual_annotation(self):
        """Create labels based on manual annotation of key frames"""
        # Based on your feedback, let's manually label the key shots
        labels = ['UNKNOWN'] * len(self.df)
        
        # Frame ranges for known shots (from your feedback):
        # Frames 15-35: Far player forehand (accurate)
        # Frames 54-84: Near player forehand (was classified as backhand, should be forehand)  
        # Frames 95-125: Far player backhand (accurate)
        # Frames 125-155: Near player backhand (accurate)
        # Frames 188-220: Far player forehand (was classified as backhand, should be forehand)
        
        # Far player (p1) shots
        for i in range(15, 36):  # Far player forehand
            if i < len(labels):
                labels[i] = 'FOREHAND_P1'
                
        for i in range(95, 126):  # Far player backhand
            if i < len(labels):
                labels[i] = 'BACKHAND_P1'
                
        for i in range(188, 221):  # Far player forehand
            if i < len(labels):
                labels[i] = 'FOREHAND_P1'
        
        # Near player (p0) shots
        for i in range(54, 85):  # Near player forehand
            if i < len(labels):
                labels[i] = 'FOREHAND_P0'
                
        for i in range(125, 156):  # Near player backhand
            if i < len(labels):
                labels[i] = 'BACKHAND_P0'
        
        self.df['manual_label'] = labels
        logger.info(f"Created manual labels. Distribution:")
        logger.info(self.df['manual_label'].value_counts())
        
        return labels
        
    def prepare_features(self):
        """Prepare features for machine learning"""
        # Select relevant features
        feature_cols = []
        
        # Ball features
        feature_cols.extend(['ball_x', 'ball_y', 'ball_confidence'])
        
        # Player 0 features
        p0_features = [col for col in self.df.columns if col.startswith('p0_')]
        feature_cols.extend(p0_features)
        
        # Player 1 features  
        p1_features = [col for col in self.df.columns if col.startswith('p1_')]
        feature_cols.extend(p1_features)
        
        # Remove any columns with all NaN values
        valid_cols = []
        for col in feature_cols:
            if col in self.df.columns and not self.df[col].isna().all():
                valid_cols.append(col)
                
        self.features = self.df[valid_cols].fillna(0)
        
        # Replace infinite values with large finite values
        self.features = self.features.replace([np.inf, -np.inf], 999999)
        logger.info(f"Selected {len(valid_cols)} features for ML")
        logger.info(f"Features: {valid_cols}")
        
        return self.features
        
    def analyze_feature_importance(self, model, feature_names):
        """Analyze feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            logger.info("Top 20 most important features:")
            for i in range(min(20, len(indices))):
                logger.info(f"{i+1:2d}. {feature_names[indices[i]]:30s} {importances[indices[i]]:.4f}")
                
            return importances, indices
        return None, None
        
    def train_models(self, X, y):
        """Train multiple ML models"""
        logger.info("Training machine learning models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            if name == 'SVM':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            logger.info(f"{name} accuracy: {accuracy:.4f}")
            
            # Feature importance for Random Forest
            if name == 'Random Forest':
                self.analyze_feature_importance(model, X.columns)
        
        self.models = results
        return results
        
    def analyze_patterns(self):
        """Analyze patterns in the data"""
        logger.info("Analyzing patterns in the data...")
        
        # Create labels
        labels = self.create_labels_from_manual_annotation()
        
        # Filter to only labeled data
        labeled_df = self.df[self.df['manual_label'] != 'UNKNOWN'].copy()
        logger.info(f"Analyzing {len(labeled_df)} labeled data points")
        
        if len(labeled_df) == 0:
            logger.warning("No labeled data found!")
            return
            
        # Analyze patterns by shot type
        shot_types = labeled_df['manual_label'].unique()
        
        for shot_type in shot_types:
            logger.info(f"\n=== {shot_type} ===")
            shot_data = labeled_df[labeled_df['manual_label'] == shot_type]
            
            # Analyze key features
            features_to_analyze = [
                'p0_arm_extension', 'p0_wrist_relative_x', 'p0_wrist_relative_y',
                'p1_arm_extension', 'p1_wrist_relative_x', 'p1_wrist_relative_y',
                'p0_ball_distance', 'p1_ball_distance'
            ]
            
            for feature in features_to_analyze:
                if feature in shot_data.columns:
                    values = shot_data[feature].dropna()
                    if len(values) > 0:
                        logger.info(f"{feature:25s}: mean={values.mean():.2f}, std={values.std():.2f}, "
                                  f"min={values.min():.2f}, max={values.max():.2f}")
        
        # Train models on labeled data
        X = self.prepare_features()
        y = labeled_df['manual_label']
        
        # Only use rows that have labels
        valid_indices = labeled_df.index
        X_labeled = X.loc[valid_indices]
        
        logger.info(f"Training on {len(X_labeled)} labeled samples")
        results = self.train_models(X_labeled, y)
        
        return results
        
    def create_visualizations(self):
        """Create visualizations of the data"""
        logger.info("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Arm extension vs wrist position
        ax1 = axes[0, 0]
        for shot_type in self.df['manual_label'].unique():
            if shot_type != 'UNKNOWN':
                data = self.df[self.df['manual_label'] == shot_type]
                ax1.scatter(data['p0_arm_extension'], data['p0_wrist_relative_x'], 
                           label=f'P0 {shot_type}', alpha=0.6)
                ax1.scatter(data['p1_arm_extension'], data['p1_wrist_relative_x'], 
                           label=f'P1 {shot_type}', alpha=0.6, marker='s')
        ax1.set_xlabel('Arm Extension')
        ax1.set_ylabel('Wrist X Position Relative to Body')
        ax1.set_title('Arm Extension vs Wrist Position')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Ball distance distribution
        ax2 = axes[0, 1]
        for shot_type in self.df['manual_label'].unique():
            if shot_type != 'UNKNOWN':
                data = self.df[self.df['manual_label'] == shot_type]
                ax2.hist(data['p0_ball_distance'], alpha=0.5, label=f'P0 {shot_type}')
                ax2.hist(data['p1_ball_distance'], alpha=0.5, label=f'P1 {shot_type}')
        ax2.set_xlabel('Ball Distance')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Ball Distance Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Wrist position scatter
        ax3 = axes[1, 0]
        for shot_type in self.df['manual_label'].unique():
            if shot_type != 'UNKNOWN':
                data = self.df[self.df['manual_label'] == shot_type]
                ax3.scatter(data['p0_wrist_relative_x'], data['p0_wrist_relative_y'], 
                           label=f'P0 {shot_type}', alpha=0.6)
                ax3.scatter(data['p1_wrist_relative_x'], data['p1_wrist_relative_y'], 
                           label=f'P1 {shot_type}', alpha=0.6, marker='s')
        ax3.set_xlabel('Wrist X Relative to Body')
        ax3.set_ylabel('Wrist Y Relative to Body')
        ax3.set_title('Wrist Position Relative to Body Center')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Arm angle distribution
        ax4 = axes[1, 1]
        for shot_type in self.df['manual_label'].unique():
            if shot_type != 'UNKNOWN':
                data = self.df[self.df['manual_label'] == shot_type]
                ax4.hist(data['p0_arm_angle'], alpha=0.5, label=f'P0 {shot_type}')
                ax4.hist(data['p1_arm_angle'], alpha=0.5, label=f'P1 {shot_type}')
        ax4.set_xlabel('Arm Angle (degrees)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Arm Angle Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tennis_shot_analysis.png', dpi=300, bbox_inches='tight')
        logger.info("Saved visualization to tennis_shot_analysis.png")
        
    def generate_insights(self):
        """Generate insights from the analysis"""
        logger.info("\n" + "="*50)
        logger.info("TENNIS SHOT CLASSIFICATION INSIGHTS")
        logger.info("="*50)
        
        # Analyze the patterns
        labeled_df = self.df[self.df['manual_label'] != 'UNKNOWN'].copy()
        
        if len(labeled_df) == 0:
            logger.warning("No labeled data to analyze!")
            return
            
        logger.info(f"\nAnalyzed {len(labeled_df)} labeled shots:")
        logger.info(labeled_df['manual_label'].value_counts())
        
        # Key insights
        logger.info("\nKEY INSIGHTS:")
        
        # 1. Arm extension patterns
        forehand_p0 = labeled_df[labeled_df['manual_label'] == 'FOREHAND_P0']
        backhand_p0 = labeled_df[labeled_df['manual_label'] == 'BACKHAND_P0']
        forehand_p1 = labeled_df[labeled_df['manual_label'] == 'FOREHAND_P1']
        backhand_p1 = labeled_df[labeled_df['manual_label'] == 'BACKHAND_P1']
        
        if len(forehand_p0) > 0 and len(backhand_p0) > 0:
            logger.info(f"\n1. NEAR PLAYER (P0) ARM EXTENSION:")
            logger.info(f"   Forehand: {forehand_p0['p0_arm_extension'].mean():.1f} ± {forehand_p0['p0_arm_extension'].std():.1f}")
            logger.info(f"   Backhand: {backhand_p0['p0_arm_extension'].mean():.1f} ± {backhand_p0['p0_arm_extension'].std():.1f}")
            
        if len(forehand_p1) > 0 and len(backhand_p1) > 0:
            logger.info(f"\n2. FAR PLAYER (P1) ARM EXTENSION:")
            logger.info(f"   Forehand: {forehand_p1['p1_arm_extension'].mean():.1f} ± {forehand_p1['p1_arm_extension'].std():.1f}")
            logger.info(f"   Backhand: {backhand_p1['p1_arm_extension'].mean():.1f} ± {backhand_p1['p1_arm_extension'].std():.1f}")
        
        # 2. Wrist position patterns
        if len(forehand_p0) > 0 and len(backhand_p0) > 0:
            logger.info(f"\n3. NEAR PLAYER (P0) WRIST POSITION:")
            logger.info(f"   Forehand X: {forehand_p0['p0_wrist_relative_x'].mean():.1f} ± {forehand_p0['p0_wrist_relative_x'].std():.1f}")
            logger.info(f"   Backhand X: {backhand_p0['p0_wrist_relative_x'].mean():.1f} ± {backhand_p0['p0_wrist_relative_x'].std():.1f}")
            
        if len(forehand_p1) > 0 and len(backhand_p1) > 0:
            logger.info(f"\n4. FAR PLAYER (P1) WRIST POSITION:")
            logger.info(f"   Forehand X: {forehand_p1['p1_wrist_relative_x'].mean():.1f} ± {forehand_p1['p1_wrist_relative_x'].std():.1f}")
            logger.info(f"   Backhand X: {backhand_p1['p1_wrist_relative_x'].mean():.1f} ± {backhand_p1['p1_wrist_relative_x'].std():.1f}")
        
        # 3. Ball distance patterns
        logger.info(f"\n5. BALL DISTANCE PATTERNS:")
        for shot_type in labeled_df['manual_label'].unique():
            if shot_type != 'UNKNOWN':
                data = labeled_df[labeled_df['manual_label'] == shot_type]
                p0_dist = data['p0_ball_distance'].mean()
                p1_dist = data['p1_ball_distance'].mean()
                logger.info(f"   {shot_type}: P0={p0_dist:.1f}px, P1={p1_dist:.1f}px")

def main():
    parser = argparse.ArgumentParser(description='Analyze tennis shot features with ML')
    parser.add_argument('--csv', required=True, help='Input CSV file with features')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    if not Path(args.csv).exists():
        logger.error(f"CSV file not found: {args.csv}")
        return
    
    # Analyze the data
    analyzer = TennisShotMLAnalyzer(args.csv)
    analyzer.load_data()
    results = analyzer.analyze_patterns()
    
    if args.visualize:
        analyzer.create_visualizations()
    
    analyzer.generate_insights()
    
    logger.info("Analysis completed!")

if __name__ == "__main__":
    main()
