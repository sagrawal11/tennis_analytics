#!/usr/bin/env python3
"""
Tennis Ball Bounce Detection - Model Training Script

This script trains a machine learning model for bounce detection using the prepared training data.
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class BounceModelTrainer:
    """Trainer for tennis ball bounce detection models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
    def load_training_data(self, train_file: str, test_file: str):
        """Load training and test data"""
        try:
            # Load training data
            self.train_df = pd.read_csv(train_file)
            logger.info(f"Loaded training data: {len(self.train_df)} samples")
            
            # Load test data
            self.test_df = pd.read_csv(test_file)
            logger.info(f"Loaded test data: {len(self.test_df)} samples")
            
            # Separate features and labels
            self.X_train = self.train_df.drop('is_bounce', axis=1)
            self.y_train = self.train_df['is_bounce']
            self.X_test = self.test_df.drop('is_bounce', axis=1)
            self.y_test = self.test_df['is_bounce']
            
            # Store feature names
            self.feature_names = list(self.X_train.columns)
            
            logger.info(f"Features: {len(self.feature_names)}")
            logger.info(f"Training samples: {len(self.X_train)} ({np.sum(self.y_train)} bounces)")
            logger.info(f"Test samples: {len(self.X_test)} ({np.sum(self.y_test)} bounces)")
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def train_models(self):
        """Train multiple models and compare performance"""
        logger.info("Starting model training...")
        
        # Standardize features for some models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Define models to train
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            logger.info(f"Training {name}...")
            
            try:
                # Use scaled features for logistic regression and SVM
                if name in ['logistic_regression', 'svm']:
                    model.fit(X_train_scaled, self.y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                auc_score = roc_auc_score(self.y_test, y_pred_proba)
                
                # Store model and scaler
                self.models[name] = model
                if name in ['logistic_regression', 'svm']:
                    self.scalers[name] = scaler
                
                results[name] = {
                    'auc_score': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - AUC: {auc_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        return results
    
    def evaluate_models(self, results):
        """Evaluate and compare model performance"""
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*60)
        
        best_model = None
        best_auc = 0
        
        for name, result in results.items():
            logger.info(f"\n{name.upper()}:")
            logger.info(f"AUC Score: {result['auc_score']:.4f}")
            
            # Classification report
            report = classification_report(self.y_test, result['predictions'], 
                                        target_names=['No Bounce', 'Bounce'])
            logger.info(f"\nClassification Report:\n{report}")
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, result['predictions'])
            logger.info(f"Confusion Matrix:\n{cm}")
            
            if result['auc_score'] > best_auc:
                best_auc = result['auc_score']
                best_model = name
        
        logger.info(f"\n🏆 BEST MODEL: {best_model.upper()} (AUC: {best_auc:.4f})")
        return best_model
    
    def save_models(self, output_dir: str):
        """Save trained models and feature information"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            model_file = output_path / f"{name}_model.joblib"
            joblib.dump(model, model_file)
            logger.info(f"Saved {name} model to {model_file}")
            
            # Save scaler if exists
            if name in self.scalers:
                scaler_file = output_path / f"{name}_scaler.joblib"
                joblib.dump(self.scalers[name], scaler_file)
                logger.info(f"Saved {name} scaler to {scaler_file}")
        
        # Save feature names
        feature_file = output_path / "feature_names.json"
        with open(feature_file, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"Saved feature names to {feature_file}")
        
        # Save training info
        info = {
            'feature_count': len(self.feature_names),
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'bounce_ratio_train': float(np.mean(self.y_train)),
            'bounce_ratio_test': float(np.mean(self.y_test)),
            'models_trained': list(self.models.keys())
        }
        
        info_file = output_path / "training_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved training info to {info_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train tennis ball bounce detection models')
    parser.add_argument('--train-data', default='bounce_training_data_high_quality_train.csv',
                       help='Training data CSV file')
    parser.add_argument('--test-data', default='bounce_training_data_high_quality_test.csv',
                       help='Test data CSV file')
    parser.add_argument('--output-dir', default='models',
                       help='Output directory for trained models')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = BounceModelTrainer()
        
        # Load data
        trainer.load_training_data(args.train_data, args.test_data)
        
        # Train models
        results = trainer.train_models()
        
        if not results:
            logger.error("No models were successfully trained!")
            return 1
        
        # Evaluate models
        best_model = trainer.evaluate_models(results)
        
        # Save models
        trainer.save_models(args.output_dir)
        
        logger.info(f"\n🎉 Training completed successfully!")
        logger.info(f"Best model: {best_model}")
        logger.info(f"Models saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
