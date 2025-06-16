"""
Data preprocessing utilities for plant disease detection.
"""

import os
import shutil
import random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing and analysis utilities."""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.class_distribution = {}
        
    def analyze_dataset(self):
        """Analyze the dataset structure and distribution."""
        logger.info("Analyzing dataset...")
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist!")
        
        # Get class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        # Count images per class
        class_counts = {}
        total_images = 0
        
        for class_dir in class_dirs:
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            class_counts[class_dir.name] = len(image_files)
            total_images += len(image_files)
        
        self.class_distribution = class_counts
        
        logger.info(f"Total classes: {len(class_counts)}")
        logger.info(f"Total images: {total_images}")
        logger.info(f"Average images per class: {total_images / len(class_counts):.1f}")
        
        # Find imbalanced classes
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count
        
        logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 5:
            logger.warning("Significant class imbalance detected!")
        
        return class_counts
    
    def plot_class_distribution(self, save_path=None):
        """Plot class distribution."""
        if not self.class_distribution:
            self.analyze_dataset()
        
        plt.figure(figsize=(15, 8))
        
        # Sort classes by count
        sorted_classes = sorted(self.class_distribution.items(), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_classes)
        
        plt.bar(range(len(classes)), counts)
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution in Dataset')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_balanced_dataset(self, output_dir, min_samples_per_class=500):
        """Create a balanced dataset by augmentation or sampling."""
        logger.info("Creating balanced dataset...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.class_distribution:
            self.analyze_dataset()
        
        for class_name, count in self.class_distribution.items():
            class_input_dir = self.data_dir / class_name
            class_output_dir = output_dir / class_name
            class_output_dir.mkdir(exist_ok=True)
            
            # Get all image files
            image_files = list(class_input_dir.glob("*.jpg")) + \
                         list(class_input_dir.glob("*.jpeg")) + \
                         list(class_input_dir.glob("*.png"))
            
            if count >= min_samples_per_class:
                # Randomly sample if too many images
                selected_files = random.sample(image_files, min_samples_per_class)
            else:
                # Use all images and augment if needed
                selected_files = image_files
                
            # Copy selected files
            for i, img_file in enumerate(selected_files):
                shutil.copy2(img_file, class_output_dir / f"{class_name}_{i:04d}{img_file.suffix}")
            
            # Augment if needed
            if count < min_samples_per_class:
                self.augment_class(class_input_dir, class_output_dir, 
                                 target_count=min_samples_per_class, 
                                 existing_count=len(selected_files))
        
        logger.info(f"Balanced dataset created in {output_dir}")
    
    def augment_class(self, input_dir, output_dir, target_count, existing_count):
        """Augment a specific class to reach target count."""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        needed_images = target_count - existing_count
        
        if needed_images <= 0:
            return
        
        logger.info(f"Augmenting {input_dir.name}: generating {needed_images} additional images")
        
        # Data augmentation generator
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Get source images
        image_files = list(input_dir.glob("*.jpg")) + \
                     list(input_dir.glob("*.jpeg")) + \
                     list(input_dir.glob("*.png"))
        
        generated_count = 0
        
        while generated_count < needed_images:
            # Randomly select source image
            source_img_path = random.choice(image_files)
            
            try:
                # Load and preprocess image
                img = Image.open(source_img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Generate augmented image
                aug_iter = datagen.flow(img_array, batch_size=1)
                aug_img = next(aug_iter)[0].astype(np.uint8)
                
                # Save augmented image
                aug_img_pil = Image.fromarray(aug_img)
                output_path = output_dir / f"{input_dir.name}_aug_{generated_count:04d}.jpg"
                aug_img_pil.save(output_path, quality=95)
                
                generated_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to augment {source_img_path}: {e}")
                continue
    
    def validate_images(self):
        """Validate all images in the dataset."""
        logger.info("Validating images...")
        
        corrupted_files = []
        valid_count = 0
        
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            for img_file in class_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    try:
                        with Image.open(img_file) as img:
                            img.verify()  # Verify image integrity
                        valid_count += 1
                    except Exception as e:
                        logger.warning(f"Corrupted image: {img_file}")
                        corrupted_files.append(img_file)
        
        logger.info(f"Valid images: {valid_count}")
        logger.info(f"Corrupted images: {len(corrupted_files)}")
        
        return corrupted_files
    
    def remove_corrupted_images(self):
        """Remove corrupted images from dataset."""
        corrupted_files = self.validate_images()
        
        for file_path in corrupted_files:
            try:
                file_path.unlink()
                logger.info(f"Removed corrupted file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
    
    def split_dataset(self, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Split dataset into train, validation, and test sets."""
        if train_ratio + val_ratio + test_ratio != 1.0:
            raise ValueError("Split ratios must sum to 1.0")
        
        output_dir = Path(output_dir)
        
        # Create split directories
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        test_dir = output_dir / "test"
        
        for split_dir in [train_dir, val_dir, test_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Splitting dataset...")
        
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Create class directories in each split
            (train_dir / class_name).mkdir(exist_ok=True)
            (val_dir / class_name).mkdir(exist_ok=True)
            (test_dir / class_name).mkdir(exist_ok=True)
            
            # Get all image files
            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))
            
            # Shuffle files
            random.shuffle(image_files)
            
            # Calculate split indices
            total_files = len(image_files)
            train_end = int(total_files * train_ratio)
            val_end = int(total_files * (train_ratio + val_ratio))
            
            # Split files
            train_files = image_files[:train_end]
            val_files = image_files[train_end:val_end]
            test_files = image_files[val_end:]
            
            # Copy files to respective directories
            for files, target_dir in [(train_files, train_dir), 
                                    (val_files, val_dir), 
                                    (test_files, test_dir)]:
                for img_file in files:
                    shutil.copy2(img_file, target_dir / class_name / img_file.name)
            
            logger.info(f"{class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        logger.info(f"Dataset split completed. Output saved to {output_dir}")

def main():
    """Main preprocessing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Plant Disease Dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, help='Output directory for processed data')
    parser.add_argument('--action', type=str, choices=['analyze', 'balance', 'split', 'validate'], 
                       default='analyze', help='Preprocessing action to perform')
    parser.add_argument('--min_samples', type=int, default=500, help='Minimum samples per class for balancing')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.data_dir)
    
    if args.action == 'analyze':
        class_counts = preprocessor.analyze_dataset()
        preprocessor.plot_class_distribution()
        
    elif args.action == 'balance':
        if not args.output_dir:
            args.output_dir = str(Path(args.data_dir).parent / "balanced_dataset")
        preprocessor.create_balanced_dataset(args.output_dir, args.min_samples)
        
    elif args.action == 'split':
        if not args.output_dir:
            args.output_dir = str(Path(args.data_dir).parent / "split_dataset")
        preprocessor.split_dataset(args.output_dir)
        
    elif args.action == 'validate':
        corrupted_files = preprocessor.validate_images()
        if corrupted_files:
            response = input(f"Found {len(corrupted_files)} corrupted files. Remove them? (y/n): ")
            if response.lower() == 'y':
                preprocessor.remove_corrupted_images()

if __name__ == "__main__":
    main()
