import os
import torch
import numpy as np

from melanoma_trainer import MelanomaModelTrainer
from combined_trainer import CombinedModelTrainer
from utils import clear_gpu_cache

def run_transfer_learning_pipeline(config):
    """
    Run the complete transfer learning pipeline
    
    Args:
        config (dict): Configuration dictionary
    """
    # Create necessary directories
    os.makedirs(config['dataset_path'], exist_ok=True)
    
    # Step 1: Train the modified melanoma model
    if config.get('train_melanoma', True):
        print("\n========== PHASE 1: TRAINING MODIFIED MELANOMA MODEL ==========\n")

        clear_gpu_cache()
        
        # Create the melanoma model trainer
        melanoma_trainer = MelanomaModelTrainer(
            dataset_path=config['dataset_path'],
            csv_file=config['csv_file'],
            img_dir=config['img_dir'],
            batch_size=config['melanoma_batch_size'],
            num_workers=config['num_workers'],
            epochs=config['melanoma_epochs'],
            learning_rate=config['melanoma_lr']
        )
        
        # If pretrained original melanoma model exists, load its weights
        if os.path.exists(config['melanoma_model_path']):
            print(f"Loading weights from melanoma model: {config['melanoma_model_path']}")
            melanoma_trainer.load_pretrained_weights(config['melanoma_model_path'])
        
        # Train the modified melanoma model
        trained_melanoma_model, melanoma_results = melanoma_trainer.train_model()
        
        # Set the path to the trained model for the next step
        melanoma_model_path = os.path.join("trained_model", f"feature_extractor_melanoma_classifier_test{config['test_num']}", "modified_melanoma_model.pth")

        clear_gpu_cache()

    else:
        # Use existing modified melanoma model
        melanoma_model_path = config['melanoma_model_path']
        print(f"Skipping melanoma model training, using existing model: {melanoma_model_path}")
        # Initialize with None since we don't have results if skipping 
        melanoma_results = None
    
    # Step 2: Train the combined transfer learning model
    if config.get('train_combined', True):
        print("\n========== PHASE 2: TRAINING COMBINED TRANSFER LEARNING MODEL ==========\n")

        clear_gpu_cache()
        
        # Create the combined model trainer
        combined_trainer = CombinedModelTrainer(
            dataset_path=config['dataset_path'],
            csv_file=config['csv_file'],
            img_dir=config['img_dir'],
            batch_size=config['combined_batch_size'],
            num_workers=config['num_workers'],
            skin_model_path=config['skin_model_path'],
            melanoma_model_path=melanoma_model_path,
            epochs=config['combined_epochs'],
            learning_rate=config['combined_lr']
        )
        
        # Train the combined model
        trained_combined_model, combined_results = combined_trainer.train_model()
        
        # Print final results
        print("\n========== FINAL RESULTS ==========\n")
        
        if melanoma_results:
            print("Modified Melanoma Model Results:")
            for key, value in melanoma_results.items():
                if key != 'confusion_matrix':
                    print(f"  {key}: {value:.4f}")
        
        print("\nCombined Transfer Learning Model Results:")
        for key, value in combined_results.items():
            if key != 'confusion_matrix':
                print(f"  {key}: {value:.4f}")
        
        if melanoma_results:
            print("\nPerformance Improvement:")
            for key in melanoma_results:
                if key not in ['confusion_matrix', 'test_loss']:
                    diff = combined_results[key] - melanoma_results[key]
                    print(f"  {key}: {diff:.4f} ({'+' if diff > 0 else ''}{diff/melanoma_results[key]*100:.2f}%)")

        clear_gpu_cache()
    
    else:
        # Skip combined model training
        print(f"Skipping combined model training")
    
    print("\n========== TRANSFER LEARNING PIPELINE COMPLETE ==========\n")