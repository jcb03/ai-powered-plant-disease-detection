"""
Image processing utilities for plant disease detection.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model prediction - EXACTLY as done during training.
    
    Args:
        image_bytes: Raw image bytes
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed image array ready for model prediction
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # CRITICAL: Direct resize WITHOUT padding (this matches training)
        # Your training used ImageDataGenerator which does direct resize
        image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # CRITICAL: Normalize pixel values to [0, 1] (matches training rescale=1./255)
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

def preprocess_image_with_padding(image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Alternative preprocessing with padding (keep for comparison).
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize with padding (your current method)
        image = resize_with_padding(image, target_size)
        
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image with padding: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

def resize_with_padding(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Resize image while maintaining aspect ratio and adding padding if necessary.
    """
    # Calculate scaling factor
    scale = min(target_size[0] / image.width, target_size[1] / image.height)
    
    # Calculate new size
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    
    # Resize image
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create new image with target size and paste resized image
    new_image = Image.new('RGB', target_size, (128, 128, 128))  # Gray padding
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    return new_image

def enhance_image_quality(image_bytes: bytes) -> bytes:
    """
    Enhance image quality for better prediction accuracy.
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply MILD enhancement techniques (don't over-enhance)
        # 1. Slight brightness adjustment
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.05)  # Reduced from 1.1
        
        # 2. Mild contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)   # Reduced from 1.2
        
        # 3. Slight color saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)  # Reduced from 1.1
        
        # 4. Mild sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.05)  # Reduced from 1.1
        
        # Convert back to bytes
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=95)
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return image_bytes  # Return original if enhancement fails

def enhance_image_opencv(image_bytes: bytes) -> bytes:
    """
    Advanced image enhancement using OpenCV.
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        # Apply MILD enhancement techniques
        # 1. Mild bilateral filter for noise reduction
        image = cv2.bilateralFilter(image, 5, 50, 50)  # Reduced intensity
        
        # 2. Mild CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Reduced clip limit
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.jpg', enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()
        
    except Exception as e:
        logger.error(f"Error in OpenCV enhancement: {str(e)}")
        return image_bytes

def validate_plant_image(image_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate if the uploaded image is likely a plant image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Check image size
        if image.size[0] < 100 or image.size[1] < 100:
            return False, "Image too small. Please upload an image at least 100x100 pixels."
        
        # Check if image is too large
        if image.size[0] > 4000 or image.size[1] > 4000:
            return False, "Image too large. Please upload an image smaller than 4000x4000 pixels."
        
        # Convert to RGB for analysis
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        # Basic color analysis for plant detection
        mean_red = np.mean(image_array[:, :, 0])
        mean_green = np.mean(image_array[:, :, 1])
        mean_blue = np.mean(image_array[:, :, 2])
        
        # Calculate green ratio (plants typically have higher green content)
        total_intensity = mean_red + mean_green + mean_blue
        if total_intensity > 0:
            green_ratio = mean_green / total_intensity
        else:
            green_ratio = 0
        
        # More lenient green content check
        if green_ratio < 0.2:  # Reduced from 0.25
            return False, "Image may not contain plant material. Please ensure the image shows plant leaves or foliage."
        
        # Check for image quality (not too dark or too bright)
        brightness = np.mean(image_array)
        if brightness < 20:  # More lenient
            return False, "Image is too dark. Please take a photo with better lighting."
        elif brightness > 235:  # More lenient
            return False, "Image is too bright or overexposed. Please adjust lighting."
        
        return True, "Image validation passed."
        
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False, f"Error processing image: {str(e)}"

def extract_image_metadata(image_bytes: bytes) -> dict:
    """
    Extract metadata from image for analysis.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        metadata = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.size[0],
            "height": image.size[1],
            "file_size": len(image_bytes),
            "aspect_ratio": round(image.size[0] / image.size[1], 2) if image.size[1] > 0 else 0
        }
        
        # Extract EXIF data if available
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            if exif:
                metadata["has_exif"] = True
                # Add relevant EXIF data
                if 272 in exif:  # Make
                    metadata["camera_make"] = exif[272]
                if 271 in exif:  # Model
                    metadata["camera_model"] = exif[271]
                if 306 in exif:  # DateTime
                    metadata["datetime"] = exif[306]
        else:
            metadata["has_exif"] = False
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return {"error": str(e)}

def create_image_thumbnail(image_bytes: bytes, size: Tuple[int, int] = (150, 150)) -> bytes:
    """
    Create a thumbnail of the image.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Create thumbnail
        image.thumbnail(size, Image.LANCZOS)
        
        # Convert back to bytes
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=85)
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error creating thumbnail: {str(e)}")
        return image_bytes

# Debug function to test preprocessing
def debug_preprocessing(image_bytes: bytes):
    """
    Debug function to compare preprocessing methods.
    """
    try:
        # Test both methods
        direct_resize = preprocess_image(image_bytes)
        with_padding = preprocess_image_with_padding(image_bytes)
        
        logger.info(f"Direct resize shape: {direct_resize.shape}")
        logger.info(f"With padding shape: {with_padding.shape}")
        logger.info(f"Direct resize range: [{direct_resize.min():.3f}, {direct_resize.max():.3f}]")
        logger.info(f"With padding range: [{with_padding.min():.3f}, {with_padding.max():.3f}]")
        
        return direct_resize, with_padding
        
    except Exception as e:
        logger.error(f"Debug preprocessing failed: {str(e)}")
        return None, None
