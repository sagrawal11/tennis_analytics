#!/usr/bin/env python3
"""
Tennis ML Trainer
Trains advanced machine learning models on annotated tennis shot data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TennisMLTrainer:
    def __init__(self, training_data_path: str):
        self.training_data_path = training_data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        
    def load_data(self):
        """Load and prepare training data"""
        logger.info(f"Loading training data from {self.training_data_path}")
        self.df = pd.read_csv(self.training_data_path)
        
        # Replace infinite values
        self.df = self.df.replace([np.inf, -np.inf], 999999)
        
        logger.info(f"Loaded {len(self.df)} training samples")
        logger.info(f"Data shape: {self.df.shape}")
        
        # Show distribution
        logger.info("Shot type distribution:")
        logger.info(self.df['true_shot_type'].value_counts())
        
        logger.info("Player distribution:")
        logger.info(self.df['player_id'].value_counts())
        
    def prepare_features(self):
        """Prepare features for training"""
        # Select feature columns (exclude metadata)
        exclude_cols = ['video_file', 'frame_number', 'player_id', 'true_shot_type', 
                       'ball_x', 'ball_y', 'ball_confidence']
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        # Remove columns with all NaN values
        valid_cols = []
        for col in feature_cols:
            if not self.df[col].isna().all():
                valid_cols.append(col)
        
        self.X = self.df[valid_cols].fillna(0)
        self.y = self.label_encoder.fit_transform(self.df['true_shot_type'])
        
        logger.info(f"Selected {len(valid_cols)} features for training")
        logger.info(f"Features: {valid_cols}")
        
        # Show feature importance preview
        logger.info(f"Feature matrix shape: {self.X.shape}")
        logger.info(f"Target shape: {self.y.shape}")
        
    def train_models(self):
        """Train multiple ML models"""
        logger.info("Training machine learning models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
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
            
            # Cross-validation
            if name == 'SVM':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            logger.info(f"{name} CV scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = results
        
        # Find best model
        best_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.best_model = results[best_name]['model']
        
        logger.info(f"Best model: {best_name} with accuracy {results[best_name]['accuracy']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model"""
        logger.info("Performing hyperparameter tuning...")
        
        # Use Random Forest for tuning (usually performs well)
        rf = RandomForestClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(self.X, self.y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def analyze_feature_importance(self):
        """Analyze feature importance"""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            logger.info("Top 20 most important features:")
            for i in range(min(20, len(indices))):
                logger.info(f"{i+1:2d}. {self.feature_names[indices[i]]:30s} {importances[indices[i]]:.4f}")
            
            return importances, indices
        return None, None
    
    def create_visualizations(self):
        """Create visualizations of the results"""
        logger.info("Creating visualizations...")
        
        # Feature importance plot
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            
            plt.figure(figsize=(12, 8))
            plt.title("Top 20 Feature Importances")
            plt.bar(range(20), importances[indices])
            plt.xticks(range(20), [self.feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            logger.info("Saved feature importance plot to feature_importance.png")
        
        # Confusion matrix for best model
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['accuracy'])
        y_test = self.models[best_model_name]['y_test']
        y_pred = self.models[best_model_name]['y_pred']
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        logger.info("Saved confusion matrix to confusion_matrix.png")
    
    def save_model(self, output_path: str):
        """Save the trained model and preprocessing objects"""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, output_path)
        logger.info(f"Saved trained model to {output_path}")
    
    def generate_insights(self):
        """Generate insights from the training"""
        logger.info("\n" + "="*60)
        logger.info("TENNIS SHOT CLASSIFICATION ML INSIGHTS")
        logger.info("="*60)
        
        # Model performance
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['accuracy'])
        best_accuracy = self.models[best_model_name]['accuracy']
        
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"Accuracy: {best_accuracy:.4f}")
        
        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            top_features = np.argsort(importances)[::-1][:10]
            
            logger.info(f"\nTop 10 Most Important Features:")
            for i, idx in enumerate(top_features):
                logger.info(f"{i+1:2d}. {self.feature_names[idx]:30s} {importances[idx]:.4f}")
        
        # Class distribution
        logger.info(f"\nTraining Data Distribution:")
        logger.info(self.df['true_shot_type'].value_counts())
        
        # Player-specific performance
        logger.info(f"\nPlayer Distribution:")
        logger.info(self.df['player_id'].value_counts())

def main():
    parser = argparse.ArgumentParser(description='Train ML models on tennis shot data')
    parser.add_argument('--data', required=True, help='Training data CSV file')
    parser.add_argument('--output', default='tennis_shot_model.pkl', help='Output model file')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        logger.error(f"Training data file not found: {args.data}")
        return
    
    # Train the model
    trainer = TennisMLTrainer(args.data)
    trainer.load_data()
    trainer.prepare_features()
    results = trainer.train_models()
    
    if args.tune:
        trainer.hyperparameter_tuning()
    
    trainer.analyze_feature_importance()
    
    if args.visualize:
        trainer.create_visualizations()
    
    trainer.save_model(args.output)
    trainer.generate_insights()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
