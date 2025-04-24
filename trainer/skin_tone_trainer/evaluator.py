import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    balanced_accuracy_score, 
    precision_recall_fscore_support, 
    mean_absolute_error,
    mean_squared_error, 
    cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
from scipy.stats import spearmanr

from metrics import (
    distance_weighted_accuracy,
    off_by_one_accuracy,
    calculate_similar_classes_confusion,
    compute_ece,
    compute_brier_score,
    plot_confusion_matrix,
    plot_calibration_curve
)
from utils import save_metrics_to_file

class ModelEvaluator:
    def __init__(self, model, device=None, output_dir=None, num_classes=7):
        """
        Initialize the model evaluator.
        
        Args:
            model: PyTorch model to evaluate
            device: Device to evaluate on
            output_dir: Directory to save evaluation results
            num_classes: Number of classes
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.num_classes = num_classes
        
        # Ensure output directory exists
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on the test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_scores = []

        # Counter for class predictions
        class_predictions = {i: 0 for i in range(10)}  # Support up to 10 classes

        # For additional analysis
        unseen_confidences = []
        unseen_origins = {i: 0 for i in range(7)}
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Evaluating'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # Get probability scores
                probs = torch.softmax(outputs, dim=1)
                all_scores.append(probs.cpu().numpy())

                # Process each prediction
                for i, (pred, true_label) in enumerate(zip(preds.cpu().numpy(), labels.cpu().numpy())):
                    # Count class predictions
                    class_predictions[pred] += 1
                    
                    # Confidence analysis for unseen classes
                    if pred >= 7:  # If predicted as unseen class
                        conf = probs[i, pred].item()
                        unseen_confidences.append(conf)
                        
                        # Which true classes get confused as unseen
                        if true_label < 7:
                            unseen_origins[true_label] += 1
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        all_scores = np.vstack(all_scores)
        all_labels_np = np.array(all_labels)
        all_preds_np = np.array(all_preds)

        # Print prediction distribution
        print("\nPrediction Distribution:")
        total_predictions = len(all_preds)
        for class_idx in range(10):
            count = class_predictions[class_idx]
            percentage = (count / total_predictions) * 100
            if class_idx < 7:
                print(f"Monk Scale {class_idx+1}: {count} predictions ({percentage:.2f}%) - In training data")
            else:
                print(f"Monk Scale {class_idx+1}: {count} predictions ({percentage:.2f}%) - UNSEEN CLASS")

        # Calculate metrics
        acc = np.mean(all_preds_np == all_labels_np)
        balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np)
        
        # Calculate macro-averaged metrics
        precision_macro, recall_macro, f1_macro, _ = \
            precision_recall_fscore_support(all_labels_np, all_preds_np, average='macro', zero_division=0)
        
        # Calculate weighted metrics
        precision_weighted, recall_weighted, f1_weighted, _ = \
            precision_recall_fscore_support(all_labels_np, all_preds_np, average='weighted', zero_division=0)
        
        # Regression metrics for ordinal tasks
        mae = mean_absolute_error(all_labels_np, all_preds_np)
        mse = mean_squared_error(all_labels_np, all_preds_np)
        kappa = cohen_kappa_score(all_labels_np, all_preds_np, weights='quadratic')

        # Ordinal classification metrics
        dw_acc = distance_weighted_accuracy(all_labels_np, all_preds_np, max_distance=9)
        off_by_one_acc = off_by_one_accuracy(all_labels_np, all_preds_np)
        
        # Confusion rates between similar classes
        similar_classes = [0, 1, 2, 3]  # 0-indexed (monk scale 1-4)
        confusion_rates = calculate_similar_classes_confusion(all_labels_np, all_preds_np, similar_classes)
        
        # Spearman's rank correlation
        spearman_corr, p_value = spearmanr(all_labels_np, all_preds_np)

        # Calculate calibration metrics
        ece = compute_ece(all_scores, all_labels_np, n_bins=10)
        brier_score = compute_brier_score(all_scores, all_labels_np, n_classes=self.num_classes)

        # Only consider classes 0-6 (monk scales 1-7) that are in the training data
        valid_classes = range(7)
        
        # One-hot encode the labels for AUC calculation
        y_true_bin = label_binarize(all_labels_np, classes=valid_classes)
        
        # Get scores for only the valid classes
        valid_scores = all_scores[:, valid_classes]
        
        # Calculate ROC AUC if possible
        try:
            from sklearn.metrics import roc_auc_score, roc_curve, auc
            # Micro-average: calculate metrics globally by considering each element
            roc_auc_micro = roc_auc_score(y_true_bin, valid_scores, multi_class='ovr', average='micro')
            
            # Macro-average: calculate metrics for each label, and find their unweighted mean
            roc_auc_macro = roc_auc_score(y_true_bin, valid_scores, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            roc_auc_micro = None
            roc_auc_macro = None

        # Confusion matrix
        cm = confusion_matrix(all_labels_np, all_preds_np, labels=range(7))
        
        # Plot confusion matrix and calibration curve
        if self.output_dir:
            class_names = [f'Scale {i+1}' for i in range(7)]
            plot_confusion_matrix(cm, class_names, output_dir=self.output_dir)
            plot_calibration_curve(all_scores, all_labels_np, output_dir=self.output_dir)

        # Compile all metrics
        metrics = {
            'accuracy': acc,
            'balanced_accuracy': balanced_acc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted, 
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'mae': mae,
            'mse': mse,
            'cohens_kappa': kappa,
            'roc_auc_micro': roc_auc_micro,
            'roc_auc_macro': roc_auc_macro,
            'distance_weighted_accuracy': dw_acc,
            'off_by_one_accuracy': off_by_one_acc,
            'spearman_correlation': spearman_corr,
            'confusion_matrix': cm,
            'class_predictions': class_predictions,
            'similar_classes_confusion': confusion_rates,
            'unseen_confidences': unseen_confidences if unseen_confidences else None,
            'unseen_origins': unseen_origins,
            'expected_calibration_error': ece,
            'brier_score': brier_score,
        }
        
        # Print main metrics
        print(f'\nTest Accuracy: {acc:.4f}')
        print(f'Test Balanced Accuracy: {balanced_acc:.4f}')
        print(f'Test Macro Precision: {precision_macro:.4f}')
        print(f'Test Macro Recall: {recall_macro:.4f}')
        print(f'Test Macro F1: {f1_macro:.4f}')
        print(f'Mean Absolute Error (MAE): {mae:.4f}')
        print(f'Mean Squared Error (MSE): {mse:.4f}')
        print(f'Cohen\'s Kappa (Quadratic Weighted): {kappa:.4f}')
        if roc_auc_macro is not None:
            print(f'ROC AUC (macro-average): {roc_auc_macro:.4f}')
        print(f'Distance-weighted Accuracy: {dw_acc:.4f}')
        print(f'Off-by-one Accuracy: {off_by_one_acc:.4f}')
        print(f'Spearman Rank Correlation: {spearman_corr:.4f}')
        print(f'Expected Calibration Error: {ece:.4f}')
        print(f'Brier Score: {brier_score:.4f}')
        
        # Save metrics to file if output directory is provided
        if self.output_dir:
            metrics_file = save_metrics_to_file(
                metrics,
                model_name="TestResults", 
                output_dir=self.output_dir,
                test_size=len(test_loader.dataset)
            )
            print(f"Metrics saved to: {metrics_file}")
        
        return metrics