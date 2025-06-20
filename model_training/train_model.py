"""
Complete model training script for plant disease detection.
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

class PlantDiseaseModelTrainer:
    """Complete plant disease model trainer."""
    
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
        
        logger.info(f"Initialized trainer with data directory: {self.data_dir}")
        
    def prepare_data(self, validation_split=0.2, test_split=0.1):
        """Prepare training, validation, and test datasets."""
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist!")
        
        logger.info("Preparing datasets...")
        
        # Data augmentation for training
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],
            validation_split=validation_split + test_split
        )
        
        # Only rescaling for validation and test
        val_test_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split + test_split
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
        self.val_dataset = val_test_datagen.flow_from_directory(
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
        logger.info(f"Classes: {self.class_names[:5]}..." if len(self.class_names) > 5 else f"Classes: {self.class_names}")
        
        # Save class names
        self.save_class_names()
        
    def save_class_names(self):
        """Save class names to JSON file."""
        class_names_file = self.output_dir / "class_names.json"
        with open(class_names_file, 'w') as f:
            json.dump(self.class_names, f, indent=2)
        logger.info(f"Class names saved to {class_names_file}")
        
    def build_model(self, model_type="efficientnet"):
        """Build the CNN model using transfer learning."""
        
        logger.info(f"Building {model_type} model...")
        
        # Choose base model
        if model_type.lower() == "efficientnet":
            base_model = keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        elif model_type.lower() == "resnet":
            base_model = keras.applications.ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        elif model_type.lower() == "mobilenet":
            base_model = keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom layers
        self.model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        logger.info("Model built successfully!")
        logger.info(f"Total parameters: {self.model.count_params():,}")
        
        # Save model summary
        with open(self.output_dir / "model_summary.txt", 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    def train_model(self, epochs=50, fine_tune_epochs=20, patience=10):
        """Train the model with transfer learning and fine-tuning."""
        
        self.training_start_time = datetime.now()
        logger.info(f"Starting model training at {self.training_start_time}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                str(self.output_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                str(self.output_dir / 'training_log.csv'),
                append=True
            )
        ]
        
        # Phase 1: Initial training with frozen base
        logger.info("Phase 1: Training with frozen base model...")
        history1 = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning
        logger.info("Phase 2: Fine-tuning with unfrozen base model...")
        
        # Unfreeze base model
        self.model.layers[0].trainable = True
        
        # Use lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Continue training
        history2 = self.model.fit(
            self.train_dataset,
            epochs=fine_tune_epochs,
            validation_data=self.val_dataset,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=len(history1.history['accuracy'])
        )
        
        # Combine histories
        self.history = {}
        for key in history1.history.keys():
            self.history[key] = history1.history[key] + history2.history[key]
        
        self.training_end_time = datetime.now()
        training_duration = self.training_end_time - self.training_start_time
        logger.info(f"Training completed in {training_duration}")
        
    def evaluate_model(self):
        """Evaluate the trained model."""
        
        logger.info("Evaluating model...")
        
        # Load best model
        best_model_path = self.output_dir / 'best_model.h5'
        if best_model_path.exists():
            self.model = keras.models.load_model(str(best_model_path))
            logger.info("Loaded best model for evaluation")
        
        # Evaluate on validation set
        val_loss, val_accuracy, val_top3_accuracy = self.model.evaluate(
            self.val_dataset, verbose=1
        )
        
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation Top-3 Accuracy: {val_top3_accuracy:.4f}")
        logger.info(f"Validation Loss: {val_loss:.4f}")
        
        # Generate predictions for detailed analysis
        logger.info("Generating predictions for analysis...")
        
        # Reset validation dataset
        self.val_dataset.reset()
        
        # Get predictions
        predictions = self.model.predict(self.val_dataset, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true classes
        true_classes = self.val_dataset.classes[:len(predicted_classes)]
        
        # Classification report
        report = classification_report(
            true_classes, predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Save classification report
        with open(self.output_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("\nClassification Report Summary:")
        logger.info(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
        logger.info(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
        logger.info(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
        
        # Confusion matrix
        self.plot_confusion_matrix(true_classes, predicted_classes)
        
        # Save evaluation metrics
        eval_metrics = {
            'val_accuracy': float(val_accuracy),
            'val_top3_accuracy': float(val_top3_accuracy),
            'val_loss': float(val_loss),
            'classification_report': report,
            'training_duration': str(self.training_end_time - self.training_start_time) if self.training_start_time else None
        }
        
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        return eval_metrics
        
    def plot_confusion_matrix(self, true_classes, predicted_classes):
        """Plot and save confusion matrix."""
        
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
        
        # Save plot
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'confusion_matrix.pdf', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {self.output_dir}")
        
    def plot_training_history(self):
        """Plot and save training history."""
        
        if not self.history:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy plot
        axes[0, 0].plot(self.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(self.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(self.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-3 Accuracy plot
        if 'top_3_accuracy' in self.history:
            axes[1, 0].plot(self.history['top_3_accuracy'], label='Training Top-3 Accuracy', linewidth=2)
            axes[1, 0].plot(self.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy', linewidth=2)
            axes[1, 0].set_title('Model Top-3 Accuracy', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-3 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot (simplified)
        axes[1, 1].set_title('Training Summary', fontsize=14, fontweight='bold')
        summary_text = f"""
        Final Training Accuracy: {self.history['accuracy'][-1]:.4f}
        Final Validation Accuracy: {self.history['val_accuracy'][-1]:.4f}
        Best Validation Accuracy: {max(self.history['val_accuracy']):.4f}
        Total Epochs: {len(self.history['accuracy'])}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plots
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'training_history.pdf', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plots saved to {self.output_dir}")
        
    def save_model(self, model_name='plant_disease_model.h5'):
        """Save the trained model."""
        
        model_path = self.output_dir / model_name
        self.model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Also save to the backend directory if it exists
        backend_model_dir = Path("../backend/ml_models")
        if backend_model_dir.exists():
            backend_model_path = backend_model_dir / model_name
            self.model.save(str(backend_model_path))
            logger.info(f"Model also saved to {backend_model_path}")
            
            # Copy class names
            class_names_src = self.output_dir / "class_names.json"
            class_names_dst = backend_model_dir / "class_names.json"
            if class_names_src.exists():
                import shutil
                shutil.copy2(str(class_names_src), str(class_names_dst))
                logger.info(f"Class names copied to {class_names_dst}")
        
        # Save training configuration
        config = {
            'model_architecture': 'EfficientNetB0 + Custom Head',
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
        
        logger.info("Model training completed successfully!")

def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=20, help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (height and width)')
    parser.add_argument('--model_type', type=str, default='efficientnet', choices=['efficientnet', 'resnet', 'mobilenet'])
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        logger.error(f"Data directory {args.data_dir} not found!")
        logger.info("Please download the PlantVillage dataset and update the data directory path.")
        return
    
    # Initialize trainer
    trainer = PlantDiseaseModelTrainer(
        data_dir=args.data_dir,
        img_height=args.img_size,
        img_width=args.img_size,
        batch_size=args.batch_size
    )
    
    try:
        # Prepare data
        logger.info("Step 1: Preparing data...")
        trainer.prepare_data()
        
        # Build model
        logger.info("Step 2: Building model...")
        trainer.build_model(model_type=args.model_type)
        
        # Train model
        logger.info("Step 3: Training model...")
        trainer.train_model(
            epochs=args.epochs, 
            fine_tune_epochs=args.fine_tune_epochs
        )
        
        # Evaluate model
        logger.info("Step 4: Evaluating model...")
        trainer.evaluate_model()
        
        # Plot training history
        logger.info("Step 5: Generating plots...")
        trainer.plot_training_history()
        
        # Save model
        logger.info("Step 6: Saving model...")
        trainer.save_model()
        
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info(f"📁 All outputs saved to: {trainer.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
