#!/usr/bin/env python3
"""
Random Forest hyperparameter tuning module for Monk skin type classification.
This module handles the tuning of hyperparameters for the Random Forest model.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def tune_random_forest(train_df, val_df, output_folder, n_iter=20, cv=5, random_state=42):
    """
    Tune hyperparameters for the Random Forest classifier using provided training and validation sets.
    
    Args:
        train_df: DataFrame with training features and skin type labels
        val_df: DataFrame with validation features and skin type labels
        output_folder: Folder to save results
        n_iter: Number of iterations for RandomizedSearchCV
        cv: Number of cross-validation folds
        random_state: Random state for reproducibility
        
    Returns:
        Best model with tuned hyperparameters
    """
    print("Tuning Random Forest hyperparameters...")
    
    # Select features and target from train set
    feature_cols = [col for col in train_df.columns if col not in 
                   ['image_name', 'predicted_skin_type', 'cluster_label']]
    X_train = train_df[feature_cols].copy()
    y_train = train_df['predicted_skin_type']
    
    # Select features and target from validation set
    X_val = val_df[feature_cols].copy()
    y_val = val_df['predicted_skin_type']
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    
    # Initialize Random Forest classifier
    rf = RandomForestClassifier(random_state=random_state)
    
    # Use RandomizedSearchCV for hyperparameter tuning
    print(f"Starting randomized search with {n_iter} iterations...")
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='accuracy',
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        refit=True
    )
    
    # Fit the model on training data
    random_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = random_search.best_params_
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Get best model
    best_model = random_search.best_estimator_
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    val_report = classification_report(y_val, y_val_pred)
    print("Validation set evaluation:")
    print(val_report)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Tuned Random Forest)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_folder, 'rf_tuned_confusion_matrix.png'))
    plt.close()
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Feature Importance (Tuned Random Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'rf_tuned_feature_importance.png'))
    plt.close()
    
    # Save hyperparameter tuning results
    with open(os.path.join(output_folder, 'rf_tuning_results.txt'), 'w') as f:
        f.write("RANDOM FOREST HYPERPARAMETER TUNING RESULTS\n")
        f.write("==========================================\n\n")
        f.write("Best parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\nBest score (CV): {:.4f}\n".format(random_search.best_score_))
        f.write("\nValidation set metrics:\n")
        f.write(val_report)
    
    return best_model


def train_tuned_random_forest(train_df, output_folder, params=None, val_df=None, test_df=None):
    """
    Train a Random Forest classifier with the specified parameters.
    
    Args:
        train_df: DataFrame with training features and skin type labels
        output_folder: Folder to save results
        params: Dictionary of hyperparameters (if None, use default)
        val_df: DataFrame with validation features and skin type labels (optional)
        test_df: DataFrame with test features and skin type labels (optional)
        
    Returns:
        Trained model
    """
    print("Training tuned Random Forest model...")
    
    # Select features and target from train set
    feature_cols = [col for col in train_df.columns if col not in 
                   ['image_name', 'predicted_skin_type', 'cluster_label']]
    X_train = train_df[feature_cols].copy()
    y_train = train_df['predicted_skin_type']
    
    # Use provided parameters or defaults
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 30,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'criterion': 'gini'
        }
    
    # Initialize and train the model
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # If validation set is provided, evaluate on it
    if val_df is not None:
        X_val = val_df[feature_cols].copy()
        y_val = val_df['predicted_skin_type']
        
        val_pred = model.predict(X_val)
        val_report = classification_report(y_val, val_pred)
        print("Validation set evaluation:")
        print(val_report)
    
    # If test set is provided, evaluate on it
    if test_df is not None:
        X_test = test_df[feature_cols].copy()
        y_test = test_df['predicted_skin_type']
        
        test_pred = model.predict(X_test)
        test_report = classification_report(y_test, test_pred)
        print("Test set evaluation:")
        print(test_report)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Random Forest)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_folder, 'rf_confusion_matrix.png'))
        plt.close()
    
    # Save hyperparameters used
    with open(os.path.join(output_folder, 'rf_parameters.txt'), 'w') as f:
        f.write("RANDOM FOREST PARAMETERS\n")
        f.write("=======================\n\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
    
    return model

def evaluate_parameter(X_train, y_train, X_test, y_test, best_params, param_name, param_values, output_folder):
    """
    Evaluate the effect of different values for a specific hyperparameter.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        best_params: Dictionary of best parameters from tuning
        param_name: Name of the parameter to evaluate
        param_values: List of values to try for the parameter
        output_folder: Folder to save results
    """
    train_scores = []
    test_scores = []
    
    for value in param_values:
        # Create a copy of best parameters and update the target parameter
        params = best_params.copy()
        params[param_name] = value
        
        # Initialize and train the model
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate scores
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, train_scores, 'o-', label='Training Accuracy')
    plt.plot(param_values, test_scores, 'o-', label='Testing Accuracy')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.title(f'Effect of {param_name} on Model Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f'rf_{param_name}_evaluation.png'))
    plt.close()

if __name__ == "__main__":
    # This allows testing the tuning module independently
    parser = argparse.ArgumentParser(description="Tune Random Forest hyperparameters")
    parser.add_argument("--data", required=True, help="Path to features CSV file")
    parser.add_argument("--output", required=True, help="Path to output folder")
    parser.add_argument("--iterations", type=int, default=20, help="Number of search iterations")
    
    args = parser.parse_args()
    
    # Load data
    features_df = pd.read_csv(args.data)
    
    # Create output folder
    os.makedirs(args.output, exist_ok=True)
    
    # Tune hyperparameters
    best_model = tune_random_forest(features_df, args.output, n_iter=args.iterations)
    
    print("Tuning complete. Results saved to", args.output)