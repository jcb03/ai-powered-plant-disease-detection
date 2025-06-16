"""
Model evaluation utilities for plant disease detection.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self, model_path, class_names=None):
        self.model_path = Path(model_path)
        self.model = None
        self.class_names = class_names or []
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate_on_dataset(self, test_data_dir, batch_size=32, img_size=(224, 224)):
        """Evaluate model on test dataset."""
        logger.info("Evaluating model on test dataset...")
        
        # Create test data generator
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Update class names if not provided
        if not self.class_names:
            self.class_names = list(test_generator.class_indices.keys())
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes,
            'class_names': self.class_names
        }
    
    def generate_classification_report(self, true_classes, predicted_classes, save_path=None):
        """Generate detailed classification report."""
        report = classification_report(
            true_classes, 
            predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Convert to DataFrame for better visualization
        df_report = pd.DataFrame(report).transpose()
        
        logger.info("Classification Report:")
        logger.info(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
        logger.info(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
        logger.info(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
        
        if save_path:
            # Save as JSON
            with open(f"{save_path}_report.json", 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save as CSV
            df_report.to_csv(f"{save_path}_report.csv")
        
        return report
    
    def plot_confusion_matrix(self, true_classes, predicted_classes, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{save_path}_confusion_matrix.pdf", bbox_inches='tight')
        
        plt.show()
        
        return cm
    
    def plot_per_class_accuracy(self, true_classes, predicted_classes, save_path=None):
        """Plot per-class accuracy."""
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Calculate per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(self.class_names)), per_class_accuracy)
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, per_class_accuracy):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_per_class_accuracy.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return per_class_accuracy
    
    def plot_roc_curves(self, true_classes, predictions, save_path=None):
        """Plot ROC curves for multi-class classification."""
        # Binarize the output
        y_test_bin = label_binarize(true_classes, classes=range(len(self.class_names)))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(self.class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves for selected classes (top 10 by AUC)
        sorted_classes = sorted(range(len(self.class_names)), 
                              key=lambda i: roc_auc[i], reverse=True)[:10]
        
        plt.figure(figsize=(12, 8))
        
        for i in sorted_classes:
            plt.plot(fpr[i], tpr[i], 
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Top 10 Classes by AUC')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_roc_curves.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return roc_auc
    
    def analyze_misclassifications(self, true_classes, predicted_classes, predictions, 
                                 top_n=10, save_path=None):
        """Analyze most common misclassifications."""
        # Find misclassified samples
        misclassified_mask = true_classes != predicted_classes
        misclassified_true = true_classes[misclassified_mask]
        misclassified_pred = predicted_classes[misclassified_mask]
        misclassified_conf = np.max(predictions[misclassified_mask], axis=1)
        
        # Count misclassification pairs
        misclass_pairs = []
        for true_idx, pred_idx, conf in zip(misclassified_true, misclassified_pred, misclassified_conf):
            misclass_pairs.append((
                self.class_names[true_idx],
                self.class_names[pred_idx],
                conf
            ))
        
        # Count occurrences
        from collections import Counter
        misclass_counts = Counter([(pair[0], pair[1]) for pair in misclass_pairs])
        
        # Get top misclassifications
        top_misclass = misclass_counts.most_common(top_n)
        
        logger.info(f"Top {top_n} Misclassifications:")
        for (true_class, pred_class), count in top_misclass:
            logger.info(f"{true_class} → {pred_class}: {count} times")
        
        # Create visualization
        if top_misclass:
            true_classes_list, pred_classes_list = zip(*[pair[0] for pair in top_misclass])
            counts = [pair[1] for pair in top_misclass]
            
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(top_misclass))
            
            bars = plt.barh(y_pos, counts)
            plt.ylabel('Misclassification Pairs')
            plt.xlabel('Count')
            plt.title(f'Top {top_n} Misclassifications')
            plt.yticks(y_pos, [f"{true} → {pred}" for true, pred in 
                              [pair[0] for pair in top_misclass]])
            
            # Add value labels
            for bar, count in zip(bars, counts):
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        str(count), ha='left', va='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}_misclassifications.png", dpi=300, bbox_inches='tight')
            
            plt.show()
        
        return top_misclass
    
    def generate_comprehensive_report(self, test_data_dir, output_dir="evaluation_results"):
        """Generate comprehensive evaluation report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Generating comprehensive evaluation report...")
        
        # Evaluate on test dataset
        eval_results = self.evaluate_on_dataset(test_data_dir)
        
        # Generate all plots and reports
        base_path = output_dir / "evaluation"
        
        # Classification report
        classification_report = self.generate_classification_report(
            eval_results['true_classes'],
            eval_results['predicted_classes'],
            save_path=str(base_path)
        )
        
        # Confusion matrix
        confusion_matrix = self.plot_confusion_matrix(
            eval_results['true_classes'],
            eval_results['predicted_classes'],
            save_path=str(base_path)
        )
        
        # Per-class accuracy
        per_class_acc = self.plot_per_class_accuracy(
            eval_results['true_classes'],
            eval_results['predicted_classes'],
            save_path=str(base_path)
        )
        
        # ROC curves
        roc_auc = self.plot_roc_curves(
            eval_results['true_classes'],
            eval_results['predictions'],
            save_path=str(base_path)
        )
        
        # Misclassification analysis
        top_misclass = self.analyze_misclassifications(
            eval_results['true_classes'],
            eval_results['predicted_classes'],
            eval_results['predictions'],
            save_path=str(base_path)
        )
        
        # Save summary
        summary = {
            'test_accuracy': eval_results['test_accuracy'],
            'test_loss': eval_results['test_loss'],
            'num_classes': len(self.class_names),
            'total_test_samples': len(eval_results['true_classes']),
            'classification_report': classification_report,
            'per_class_accuracy': per_class_acc.tolist(),
            'average_roc_auc': np.mean(list(roc_auc.values())),
            'top_misclassifications': top_misclass
        }
        
        with open(output_dir / "evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Comprehensive evaluation report saved to {output_dir}")
        
        return summary

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Plant Disease Detection Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_data_dir', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--class_names_file', type=str, help='Path to class names JSON file')
    
    args = parser.parse_args()
    
    # Load class names if provided
    class_names = None
    if args.class_names_file:
        with open(args.class_names_file, 'r') as f:
            class_names = json.load(f)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, class_names)
    
    # Generate comprehensive report
    summary = evaluator.generate_comprehensive_report(args.test_data_dir, args.output_dir)
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Test Accuracy: {summary['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
