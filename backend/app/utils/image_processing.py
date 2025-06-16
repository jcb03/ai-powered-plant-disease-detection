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
    Preprocess image for model prediction.
    
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
        
        # Resize image maintaining aspect ratio
        image = resize_with_padding(image, target_size)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

def resize_with_padding(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Resize image while maintaining aspect ratio and adding padding if necessary.
    
    Args:
        image: PIL Image object
        target_size: Target size (width, height)
        
    Returns:
        Resized image with padding
    """
    # Calculate scaling factor
    scale = min(target_size[0] / image.width, target_size[1] / image.height)
    
    # Calculate new size
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    
    # Resize image
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and paste resized image
    new_image = Image.new('RGB', target_size, (0, 0, 0))
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    return new_image

def enhance_image_quality(image_bytes: bytes) -> bytes:
    """
    Enhance image quality for better prediction accuracy.
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Enhanced image bytes
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply enhancement techniques
        # 1. Brightness adjustment
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        # 2. Contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # 3. Color saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        # 4. Sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # 5. Noise reduction using PIL filter
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
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
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Enhanced image bytes
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        # Apply advanced enhancement techniques
        # 1. Bilateral filter for noise reduction while preserving edges
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Gamma correction for brightness adjustment
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.jpg', enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()
        
    except Exception as e:
        logger.error(f"Error in OpenCV enhancement: {str(e)}")
        return image_bytes  # Return original if enhancement fails

def validate_plant_image(image_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate if the uploaded image is likely a plant image.
    
    Args:
        image_bytes: Raw image bytes
    
    Returns:
        Tuple of (is_valid, message)
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
        # Calculate color channel statistics
        mean_red = np.mean(image_array[:, :, 0])
        mean_green = np.mean(image_array[:, :, 1])
        mean_blue = np.mean(image_array[:, :, 2])
        
        # Calculate green ratio (plants typically have higher green content)
        total_intensity = mean_red + mean_green + mean_blue
        if total_intensity > 0:
            green_ratio = mean_green / total_intensity
        else:
            green_ratio = 0
        
        # Check for sufficient green content
        if green_ratio < 0.25:
            return False, "Image may not contain plant material. Please ensure the image shows plant leaves or foliage."
        
        # Check for image quality (not too dark or too bright)
        brightness = np.mean(image_array)
        if brightness < 30:
            return False, "Image is too dark. Please take a photo with better lighting."
        elif brightness > 225:
            return False, "Image is too bright or overexposed. Please adjust lighting."
        
        # Check for blur (using Laplacian variance)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score < 100:
            return False, "Image appears to be blurry. Please take a clearer photo."
        
        return True, "Image validation passed."
        
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False, f"Error processing image: {str(e)}"

def extract_image_metadata(image_bytes: bytes) -> dict:
    """
    Extract metadata from image for analysis.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Dictionary containing image metadata
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
    
    Args:
        image_bytes: Raw image bytes
        size: Thumbnail size (width, height)
        
    Returns:
        Thumbnail image bytes
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Create thumbnail
        image.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Convert back to bytes
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=85)
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error creating thumbnail: {str(e)}")
        return image_bytes
