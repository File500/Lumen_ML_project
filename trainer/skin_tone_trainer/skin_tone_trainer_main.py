import os
import sys
import torch
import torch.optim as optim

# Import local modules
from config import BaseConfig
from dataset import prepare_data
from skin_tone_trainer import ModelTrainer
from evaluator import ModelEvaluator
from metrics import apply_temperature_scaling
from loss import FocalOrdinalWassersteinLoss
from utils import save_model_architecture, save_training_parameters

# Import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.skin_tone_model import EfficientNetB3SkinToneClassifier

def main():
    """
    Main function to run the skin tone classification model training and evaluation.
    """
    # Use ConfigB3 for this implementation
    config = BaseConfig()
    
    print(f"Using device: {config.DEVICE}")
    print(f"Using model type: {config.MODEL_TYPE}")
    

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Prepare data loaders
    train_loader, val_loader, test_loader, monk_scale_counts, class_weights = prepare_data(
        csv_path=config.CSV_PATH,
        image_dir=config.IMAGE_DIR,
        batch_size=config.BATCH_SIZE,
        test_batch_size=config.TEST_BATCH_SIZE,
        val_size=config.VAL_SIZE,
        test_size=config.TEST_SIZE,
        model_type=config.MODEL_TYPE,
        augmentation_method="both",
        output_dir=config.OUTPUT_DIR
    )
    
    # Create model
    model = EfficientNetB3SkinToneClassifier(
        num_classes=config.NUM_CLASSES,
        freeze_features=config.FREEZE_FEATURES
    )
    model = model.to(config.DEVICE)
    
    # Save model architecture
    save_model_architecture(
        model, 
        model_name=f"EfficientNet{config.MODEL_TYPE}", 
        output_dir=config.OUTPUT_DIR, 
        include_params=True
    )
    
    # Define loss function
    criterion = FocalOrdinalWassersteinLoss(
        num_classes=config.NUM_CLASSES,
        alpha=0.25,
        gamma=2.0,
        ordinal_weight=0.3,
        wasserstein_weight=0.4
    )
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.0001
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Save training parameters
    save_training_parameters(optimizer, criterion, scheduler, output_dir=config.OUTPUT_DIR)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        output_dir=config.OUTPUT_DIR
    )
    
    # Train model
    print("Starting model training...")
    model, history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        patience=config.PATIENCE
    )
    
    # Apply temperature scaling for calibration
    print("Calibrating model with temperature scaling...")
    calibrated_model = apply_temperature_scaling(model, val_loader, config.DEVICE)
    
    # Save the calibrated model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'temperature': calibrated_model.temperature,
    }, os.path.join(config.OUTPUT_DIR, 'calibrated_model.pth'))
    
    # Evaluate original model
    print("Evaluating original model...")
    original_evaluator = ModelEvaluator(
        model=model,
        device=config.DEVICE,
        output_dir=os.path.join(config.OUTPUT_DIR, 'original_model_results'),
        num_classes=config.NUM_CLASSES
    )
    
    original_metrics = original_evaluator.evaluate(test_loader)
    
    # Evaluate calibrated model
    print("Evaluating calibrated model...")
    calibrated_evaluator = ModelEvaluator(
        model=calibrated_model,
        device=config.DEVICE,
        output_dir=config.OUTPUT_DIR,  # Main output directory for calibrated results
        num_classes=config.NUM_CLASSES
    )
    
    calibrated_metrics = calibrated_evaluator.evaluate(test_loader)
    
    # Compare results
    print("\nComparison of models:")
    print(f"Original model:")
    print(f"  - Accuracy: {original_metrics['accuracy']:.4f}")
    print(f"  - Balanced Accuracy: {original_metrics['balanced_accuracy']:.4f}")
    print(f"  - F1 Score: {original_metrics['f1_macro']:.4f}")
    print(f"  - ECE: {original_metrics['expected_calibration_error']:.4f}")
    
    print(f"Calibrated model:")
    print(f"  - Accuracy: {calibrated_metrics['accuracy']:.4f}")
    print(f"  - Balanced Accuracy: {calibrated_metrics['balanced_accuracy']:.4f}")
    print(f"  - F1 Score: {calibrated_metrics['f1_macro']:.4f}")
    print(f"  - ECE: {calibrated_metrics['expected_calibration_error']:.4f}")
    
    # Return the better model based on balanced accuracy
    if calibrated_metrics['balanced_accuracy'] > original_metrics['balanced_accuracy']:
        print(f"Calibrated model performs better and is saved to {os.path.join(config.OUTPUT_DIR, 'calibrated_model.pth')}")
        return calibrated_model, history
    else:
        print(f"Original model performs better and is saved to {os.path.join(config.OUTPUT_DIR, 'best_model.pth')}")
        return model, history

if __name__ == "__main__":
    main()