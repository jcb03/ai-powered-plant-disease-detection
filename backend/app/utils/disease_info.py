"""
Disease information database and utilities.
"""

from typing import Dict, List, Optional

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "disease_name": "Apple Scab",
        "scientific_name": "Venturia inaequalis",
        "crop": "Apple",
        "severity": "Moderate",
        "description": "A fungal disease that affects apple trees, causing dark, scaly spots on leaves and fruit.",
        "symptoms": [
            "Dark, olive-green to black spots on leaves",
            "Scaly, rough lesions on fruit",
            "Premature leaf drop",
            "Reduced fruit quality and yield",
            "Cracked or deformed fruit"
        ],
        "causes": [
            "Fungal infection by Venturia inaequalis",
            "High humidity and moisture",
            "Poor air circulation",
            "Infected plant debris"
        ],
        "treatment": [
            "Apply fungicide sprays during wet weather",
            "Use copper-based fungicides in early spring",
            "Remove fallen leaves and debris",
            "Prune for better air circulation",
            "Apply preventive sprays before symptoms appear"
        ],
        "prevention": [
            "Plant resistant apple varieties",
            "Ensure good air circulation",
            "Avoid overhead watering",
            "Regular pruning and sanitation",
            "Apply dormant oil sprays"
        ],
        "organic_treatment": [
            "Baking soda spray (1 tsp per quart water)",
            "Neem oil application",
            "Compost tea foliar spray",
            "Milk spray (1:10 ratio with water)"
        ]
    },
    
    "Apple___Black_rot": {
        "disease_name": "Apple Black Rot",
        "scientific_name": "Botryosphaeria obtusa",
        "crop": "Apple",
        "severity": "High",
        "description": "A serious fungal disease causing black rot on apple fruit and cankers on branches.",
        "symptoms": [
            "Brown spots with concentric rings on leaves",
            "Black, mummified fruit",
            "Sunken cankers on branches",
            "Premature fruit drop",
            "Wilting of shoots"
        ],
        "causes": [
            "Fungal infection by Botryosphaeria obtusa",
            "Wounds in bark or fruit",
            "Stress conditions",
            "Poor orchard sanitation"
        ],
        "treatment": [
            "Remove infected fruit and branches immediately",
            "Apply copper-based fungicides",
            "Improve orchard sanitation",
            "Prune dead and diseased wood",
            "Apply wound dressings to cuts"
        ],
        "prevention": [
            "Regular pruning and sanitation",
            "Remove mummified fruit",
            "Apply preventive fungicides",
            "Maintain tree health and vigor",
            "Avoid mechanical injuries"
        ],
        "organic_treatment": [
            "Bordeaux mixture application",
            "Lime sulfur spray",
            "Proper sanitation practices",
            "Beneficial microorganism applications"
        ]
    },
    
    "Apple___Cedar_apple_rust": {
        "disease_name": "Cedar Apple Rust",
        "scientific_name": "Gymnosporangium juniperi-virginianae",
        "crop": "Apple",
        "severity": "Moderate",
        "description": "A fungal disease requiring both apple and cedar trees to complete its life cycle.",
        "symptoms": [
            "Bright yellow spots on upper leaf surface",
            "Orange pustules on leaf undersides",
            "Premature defoliation",
            "Reduced fruit quality",
            "Stunted growth"
        ],
        "causes": [
            "Fungal spores from cedar trees",
            "Alternating host requirement",
            "Wet spring weather",
            "Proximity to cedar/juniper trees"
        ],
        "treatment": [
            "Apply fungicides in early spring",
            "Remove nearby cedar trees if possible",
            "Use resistant apple varieties",
            "Regular monitoring and early intervention"
        ],
        "prevention": [
            "Plant resistant apple varieties",
            "Remove cedar trees within 2 miles if possible",
            "Apply preventive fungicides",
            "Monitor weather conditions",
            "Improve air circulation"
        ],
        "organic_treatment": [
            "Sulfur-based fungicides",
            "Resistant variety selection",
            "Cultural control methods",
            "Proper spacing and pruning"
        ]
    },
    
    "Apple___healthy": {
        "disease_name": "Healthy Apple",
        "scientific_name": "N/A",
        "crop": "Apple",
        "severity": "None",
        "description": "Your apple plant appears to be healthy with no visible disease symptoms!",
        "symptoms": [
            "Green, vibrant leaves",
            "No visible spots or discoloration",
            "Normal growth pattern",
            "Healthy fruit development"
        ],
        "causes": ["Plant is healthy"],
        "treatment": ["Continue regular care and monitoring"],
        "prevention": [
            "Maintain proper watering schedule",
            "Ensure adequate nutrition",
            "Regular pruning for air circulation",
            "Monitor for early signs of disease",
            "Apply preventive treatments as needed"
        ],
        "organic_treatment": [
            "Continue organic practices",
            "Maintain soil health",
            "Use beneficial insects",
            "Regular monitoring"
        ]
    },
    
    # Tomato diseases
    "Tomato___Late_blight": {
        "disease_name": "Tomato Late Blight",
        "scientific_name": "Phytophthora infestans",
        "crop": "Tomato",
        "severity": "Very High",
        "description": "A devastating disease that can destroy entire tomato crops quickly in favorable conditions.",
        "symptoms": [
            "Dark, water-soaked spots on leaves",
            "White mold growth on leaf undersides",
            "Brown, greasy spots on fruit",
            "Rapid plant collapse",
            "Foul odor from infected tissues"
        ],
        "causes": [
            "Phytophthora infestans pathogen",
            "Cool, wet weather conditions",
            "High humidity",
            "Poor air circulation",
            "Infected plant material"
        ],
        "treatment": [
            "Apply copper-based fungicides immediately",
            "Remove infected plants completely",
            "Improve air circulation",
            "Avoid overhead watering",
            "Apply systemic fungicides"
        ],
        "prevention": [
            "Use resistant varieties",
            "Ensure good air circulation",
            "Water at soil level only",
            "Apply preventive fungicides",
            "Remove plant debris"
        ],
        "organic_treatment": [
            "Copper sulfate spray",
            "Baking soda solution",
            "Proper plant spacing",
            "Mulching to prevent soil splash"
        ]
    },
    
    "Tomato___Early_blight": {
        "disease_name": "Tomato Early Blight",
        "scientific_name": "Alternaria solani",
        "crop": "Tomato",
        "severity": "Moderate",
        "description": "A common fungal disease affecting tomato plants, especially older leaves.",
        "symptoms": [
            "Brown spots with concentric rings on leaves",
            "Yellowing and dropping of lower leaves",
            "Dark, sunken spots on fruit",
            "Stem lesions near soil line",
            "Reduced yield"
        ],
        "causes": [
            "Alternaria solani fungus",
            "Warm, humid conditions",
            "Plant stress",
            "Poor nutrition",
            "Overhead watering"
        ],
        "treatment": [
            "Apply fungicides containing chlorothalonil",
            "Remove affected leaves",
            "Improve air circulation",
            "Mulch around plants",
            "Ensure proper nutrition"
        ],
        "prevention": [
            "Rotate crops annually",
            "Use resistant varieties",
            "Proper plant spacing",
            "Avoid overhead irrigation",
            "Maintain plant health"
        ],
        "organic_treatment": [
            "Neem oil spray",
            "Compost tea application",
            "Proper mulching",
            "Beneficial microorganism inoculation"
        ]
    },
    
    "Tomato___healthy": {
        "disease_name": "Healthy Tomato",
        "scientific_name": "N/A",
        "crop": "Tomato",
        "severity": "None",
        "description": "Your tomato plant looks healthy with vibrant foliage and no disease symptoms!",
        "symptoms": [
            "Green, healthy foliage",
            "No disease symptoms visible",
            "Normal growth and development",
            "Healthy fruit set"
        ],
        "causes": ["Plant is healthy"],
        "treatment": ["Continue current care routine"],
        "prevention": [
            "Maintain consistent watering",
            "Provide adequate support",
            "Monitor regularly for pests and diseases",
            "Ensure proper nutrition",
            "Maintain good air circulation"
        ],
        "organic_treatment": [
            "Continue organic gardening practices",
            "Regular soil amendments",
            "Companion planting",
            "Natural pest management"
        ]
    },
    
    # Additional crops can be added here
    "Corn_(maize)___Northern_Leaf_Blight": {
        "disease_name": "Northern Corn Leaf Blight",
        "scientific_name": "Exserohilum turcicum",
        "crop": "Corn",
        "severity": "High",
        "description": "A fungal disease causing significant yield losses in corn crops.",
        "symptoms": [
            "Long, elliptical lesions on leaves",
            "Gray-green to tan colored spots",
            "Lesions may have dark borders",
            "Premature leaf death",
            "Reduced photosynthesis"
        ],
        "causes": [
            "Exserohilum turcicum fungus",
            "Warm, humid weather",
            "Poor air circulation",
            "Infected crop residue"
        ],
        "treatment": [
            "Apply fungicides at first sign",
            "Use resistant corn varieties",
            "Improve field drainage",
            "Remove infected plant debris"
        ],
        "prevention": [
            "Crop rotation",
            "Use resistant hybrids",
            "Proper field sanitation",
            "Balanced fertilization"
        ],
        "organic_treatment": [
            "Biological control agents",
            "Crop rotation with non-host plants",
            "Organic fungicide applications",
            "Soil health improvement"
        ]
    }
}

def get_disease_info(prediction_class: str) -> Dict:
    """
    Get detailed information about a predicted disease class.
    
    Args:
        prediction_class: The predicted disease class name
        
    Returns:
        Dictionary containing disease information
    """
    return DISEASE_INFO.get(prediction_class, {
        "disease_name": "Unknown Disease",
        "scientific_name": "Unknown",
        "crop": "Unknown",
        "severity": "Unknown",
        "description": "Disease information not available in our database.",
        "symptoms": ["Consult with agricultural expert for proper diagnosis"],
        "causes": ["Unknown pathogen or condition"],
        "treatment": ["Seek professional agricultural advice"],
        "prevention": ["Regular monitoring and good agricultural practices recommended"],
        "organic_treatment": ["Consult organic farming specialists"]
    })

def get_diseases_by_crop(crop_name: str) -> List[Dict]:
    """
    Get all diseases for a specific crop.
    
    Args:
        crop_name: Name of the crop
        
    Returns:
        List of disease information dictionaries
    """
    diseases = []
    for class_name, info in DISEASE_INFO.items():
        if info.get("crop", "").lower() == crop_name.lower():
            diseases.append({
                "class_name": class_name,
                **info
            })
    return diseases

def get_severity_color(severity: str) -> str:
    """
    Get color code for disease severity level.
    
    Args:
        severity: Severity level string
        
    Returns:
        Color code for UI display
    """
    severity_colors = {
        "None": "#4CAF50",      # Green
        "Low": "#8BC34A",       # Light Green
        "Moderate": "#FF9800",  # Orange
        "High": "#FF5722",      # Red Orange
        "Very High": "#F44336"  # Red
    }
    return severity_colors.get(severity, "#9E9E9E")  # Gray for unknown

def search_diseases(query: str) -> List[Dict]:
    """
    Search diseases by name, symptoms, or crop.
    
    Args:
        query: Search query string
        
    Returns:
        List of matching disease information
    """
    query = query.lower()
    results = []
    
    for class_name, info in DISEASE_INFO.items():
        # Search in disease name, crop, symptoms, and description
        searchable_text = " ".join([
            info.get("disease_name", ""),
            info.get("crop", ""),
            info.get("description", ""),
            " ".join(info.get("symptoms", []))
        ]).lower()
        
        if query in searchable_text:
            results.append({
                "class_name": class_name,
                **info
            })
    
    return results
