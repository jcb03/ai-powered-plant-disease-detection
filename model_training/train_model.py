"""
Complete model training script for plant disease detection - NO FINE-TUNING VERSION.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import argparse
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class HighAccuracyPlantDiseaseTrainer:
    """High accuracy trainer WITHOUT fine-tuning to avoid errors."""
    
    def __init__(self, data_dir, img_height=224, img_width=224, batch_size=32):
        self.data_dir = Path(data_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        self.num_classes = 0
        
        # Create output directories
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Training metadata
        self.training_start_time = None
        self.training_end_time = None
        
        logger.info(f"Initialized HIGH ACCURACY trainer (NO FINE-TUNING) with data directory: {self.data_dir}")
        
    def prepare_data(self, validation_split=0.1):
        """Prepare datasets with minimal augmentation for faster convergence."""
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist!")
        
        logger.info("Preparing datasets with MINIMAL augmentation...")
        
        # MINIMAL augmentation for faster convergence
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training data
        self.train_dataset = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation data
        self.val_dataset = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Get class information
        self.class_names = list(self.train_dataset.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        logger.info(f"Training samples: {self.train_dataset.samples}")
        logger.info(f"Validation samples: {self.val_dataset.samples}")
        logger.info(f"Number of classes: {self.num_classes}")
        
        # Save class names
        self.save_class_names()
        
    def save_class_names(self):
        """Save class names to JSON file."""
        class_names_file = self.output_dir / "class_names.json"
        with open(class_names_file, 'w') as f:
            json.dump(self.class_names, f, indent=2)
        logger.info(f"Class names saved to {class_names_file}")
        
    def build_model(self, model_type="resnet"):
        """Build model with ResNet50V2 backbone for superior performance."""
        
        logger.info(f"Building {model_type} model for HIGH ACCURACY (NO FINE-TUNING)...")
        
        # Use ResNet50V2 - proven to work better than EfficientNet for this task
        if model_type.lower() == "resnet":
            base_model = keras.applications.ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        elif model_type.lower() == "efficientnet":
            base_model = keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # SIMPLIFIED architecture with less dropout for better learning
        self.model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # HIGHER learning rate for initial training
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("HIGH ACCURACY model built successfully!")
        logger.info(f"Total parameters: {self.model.count_params():,}")
        
        # Save model summary
        with open(self.output_dir / "model_summary.txt", 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    def train_model(self, epochs=40, patience=15):
        """Train with single phase - NO FINE-TUNING to avoid errors."""
        
        self.training_start_time = datetime.now()
        logger.info(f"Starting HIGH ACCURACY training (NO FINE-TUNING) at {self.training_start_time}")
        
        # AGGRESSIVE callbacks for high performance
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=5,
                min_lr=1e-8,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                str(self.output_dir / 'best_model_weights.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]
        
        # SINGLE PHASE: AGGRESSIVE training with frozen base model
        logger.info("SINGLE PHASE: AGGRESSIVE training with frozen base model...")
        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store history directly (no combination needed)
        self.history = history.history
        
        self.training_end_time = datetime.now()
        training_duration = self.training_end_time - self.training_start_time
        logger.info(f"HIGH ACCURACY training (NO FINE-TUNING) completed in {training_duration}")
        
    def evaluate_model(self):
        """Evaluate the trained model."""
        
        logger.info("Evaluating HIGH ACCURACY model...")
        
        # Load best weights
        weights_path = self.output_dir / 'best_model_weights.h5'
        
        if weights_path.exists():
            try:
                self.model.load_weights(str(weights_path))
                logger.info("Loaded best model weights for evaluation")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}. Using current model.")
        
        # Evaluate on validation set
        try:
            eval_results = self.model.evaluate(self.val_dataset, verbose=1)
            
            if isinstance(eval_results, list) and len(eval_results) >= 2:
                val_loss, val_accuracy = eval_results[0], eval_results[1]
                logger.info(f"FINAL Validation Loss: {val_loss:.4f}")
                logger.info(f"FINAL Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.1f}%)")
            else:
                val_loss = eval_results
                val_accuracy = 0.0
                logger.info(f"FINAL Validation Loss: {val_loss:.4f}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
        
        # Generate predictions for detailed analysis
        logger.info("Generating predictions for analysis...")
        
        try:
            self.val_dataset.reset()
            predictions = self.model.predict(self.val_dataset, verbose=1)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = self.val_dataset.classes[:len(predicted_classes)]
            
            # Classification report
            report = classification_report(
                true_classes, predicted_classes,
                target_names=self.class_names,
                output_dict=True
            )
            
            with open(self.output_dir / 'classification_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("\nHIGH ACCURACY Classification Report Summary:")
            logger.info(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
            logger.info(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
            logger.info(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
            
            self.plot_confusion_matrix(true_classes, predicted_classes)
            
        except Exception as e:
            logger.error(f"Prediction analysis failed: {e}")
        
        # Save evaluation metrics
        eval_metrics = {
            'val_accuracy': float(val_accuracy),
            'val_loss': float(val_loss),
            'training_duration': str(self.training_end_time - self.training_start_time) if self.training_start_time else None
        }
        
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        return eval_metrics
        
    def plot_confusion_matrix(self, true_classes, predicted_classes):
        """Plot and save confusion matrix."""
        
        try:
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
            plt.title('HIGH ACCURACY Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=14)
            plt.ylabel('True Label', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"HIGH ACCURACY confusion matrix saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create confusion matrix: {e}")
        
    def plot_training_history(self):
        """Plot and save training history."""
        
        if not self.history:
            logger.warning("No training history available")
            return
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy plot
            axes[0].plot(self.history['accuracy'], label='Training Accuracy', linewidth=2)
            axes[0].plot(self.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
            axes[0].set_title('HIGH ACCURACY Model Performance (NO FINE-TUNING)', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Loss plot
            axes[1].plot(self.history['loss'], label='Training Loss', linewidth=2)
            axes[1].plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
            axes[1].set_title('HIGH ACCURACY Model Loss (NO FINE-TUNING)', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"HIGH ACCURACY training history plots saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create training history plots: {e}")
        
    def save_model(self, model_name='plant_disease_model.h5'):
        """Save the trained model."""
        
        try:
            model_path = self.output_dir / model_name
            
            # Save model without optimizer
            try:
                self.model.save(str(model_path), save_format='h5', include_optimizer=False)
                logger.info(f"HIGH ACCURACY model saved to {model_path}")
            except Exception as e:
                logger.warning(f"Full model save failed: {e}")
                # Save weights as fallback
                weights_path = str(model_path).replace('.h5', '_weights.h5')
                self.model.save_weights(weights_path)
                logger.info(f"HIGH ACCURACY model weights saved to {weights_path}")
            
            # Save to backend directory
            backend_model_dir = Path("../backend/ml_models")
            if backend_model_dir.exists():
                backend_model_path = backend_model_dir / model_name
                try:
                    self.model.save(str(backend_model_path), save_format='h5', include_optimizer=False)
                    logger.info(f"HIGH ACCURACY model also saved to {backend_model_path}")
                except Exception as e:
                    logger.warning(f"Backend model save failed: {e}")
                
                # Copy class names
                class_names_src = self.output_dir / "class_names.json"
                class_names_dst = backend_model_dir / "class_names.json"
                if class_names_src.exists():
                    import shutil
                    shutil.copy2(str(class_names_src), str(class_names_dst))
                    logger.info(f"Class names copied to {class_names_dst}")
            
            # Save training configuration
            config = {
                'model_architecture': 'ResNet50V2 + Custom Head (HIGH ACCURACY - NO FINE-TUNING)',
                'input_shape': [self.img_height, self.img_width, 3],
                'num_classes': self.num_classes,
                'batch_size': self.batch_size,
                'total_parameters': self.model.count_params(),
                'class_names': self.class_names,
                'training_start_time': self.training_start_time.isoformat() if self.training_start_time else None,
                'training_end_time': self.training_end_time.isoformat() if self.training_end_time else None
            }
            
            with open(self.output_dir / 'model_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("HIGH ACCURACY model training (NO FINE-TUNING) completed successfully!")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description='Train HIGH ACCURACY Plant Disease Detection Model (NO FINE-TUNING)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (height and width)')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'efficientnet'])
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory {args.data_dir} not found!")
        logger.info("Please download the PlantVillage dataset and update the data directory path.")
        return
    
    # Initialize HIGH ACCURACY trainer
    trainer = HighAccuracyPlantDiseaseTrainer(
        data_dir=args.data_dir,
        img_height=args.img_size,
        img_width=args.img_size,
        batch_size=args.batch_size
    )
    
    try:
        # Prepare data
        logger.info("Step 1: Preparing data for HIGH ACCURACY (NO FINE-TUNING)...")
        trainer.prepare_data()
        
        # Build model
        logger.info("Step 2: Building HIGH ACCURACY model (NO FINE-TUNING)...")
        trainer.build_model(model_type=args.model_type)
        
        # Train model
        logger.info("Step 3: Training HIGH ACCURACY model (NO FINE-TUNING)...")
        trainer.train_model(epochs=args.epochs)
        
        # Evaluate model
        logger.info("Step 4: Evaluating HIGH ACCURACY model...")
        trainer.evaluate_model()
        
        # Plot training history
        logger.info("Step 5: Generating HIGH ACCURACY plots...")
        trainer.plot_training_history()
        
        # Save model
        logger.info("Step 6: Saving HIGH ACCURACY model...")
        trainer.save_model()
        
        logger.info("🎉 HIGH ACCURACY training pipeline (NO FINE-TUNING) completed successfully!")
        logger.info(f"📁 All outputs saved to: {trainer.output_dir}")
        
    except Exception as e:
        logger.error(f"HIGH ACCURACY training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
