"""
Disease information database and utilities.
"""

from typing import Dict, List, Optional

# Complete disease information database
DISEASE_INFO = {
    # Apple diseases
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
    
    # BLUEBERRY DISEASES
    "Blueberry___healthy": {
        "disease_name": "Healthy Blueberry",
        "scientific_name": "N/A",
        "crop": "Blueberry",
        "severity": "None",
        "description": "Your blueberry plant appears healthy with vibrant foliage and no disease symptoms!",
        "symptoms": [
            "Green, healthy leaves",
            "No visible disease symptoms",
            "Normal growth and berry development",
            "Good plant vigor"
        ],
        "causes": ["Plant is healthy"],
        "treatment": ["Continue current care practices"],
        "prevention": [
            "Maintain proper soil pH (4.5-5.5)",
            "Ensure adequate drainage",
            "Regular pruning for air circulation",
            "Monitor for pests and diseases",
            "Proper fertilization"
        ],
        "organic_treatment": [
            "Continue organic practices",
            "Maintain acidic soil conditions",
            "Use organic mulch",
            "Regular monitoring"
        ]
    },
    
    # CHERRY DISEASES
    "Cherry_(including_sour)___Powdery_mildew": {
        "disease_name": "Cherry Powdery Mildew",
        "scientific_name": "Podosphaera clandestina",
        "crop": "Cherry",
        "severity": "Moderate",
        "description": "A fungal disease causing white powdery growth on cherry leaves and shoots.",
        "symptoms": [
            "White powdery coating on leaves",
            "Distorted and curled leaves",
            "Stunted shoot growth",
            "Premature leaf drop",
            "Reduced fruit quality"
        ],
        "causes": [
            "Podosphaera clandestina fungus",
            "High humidity with dry conditions",
            "Poor air circulation",
            "Overcrowded plantings"
        ],
        "treatment": [
            "Apply sulfur-based fungicides",
            "Use systemic fungicides like myclobutanil",
            "Improve air circulation",
            "Remove infected plant material",
            "Apply horticultural oils"
        ],
        "prevention": [
            "Plant resistant cherry varieties",
            "Ensure proper spacing",
            "Prune for good air circulation",
            "Avoid overhead watering",
            "Regular monitoring"
        ],
        "organic_treatment": [
            "Sulfur dust or spray",
            "Baking soda solution",
            "Neem oil application",
            "Proper pruning practices"
        ]
    },
    
    "Cherry_(including_sour)___healthy": {
        "disease_name": "Healthy Cherry",
        "scientific_name": "N/A",
        "crop": "Cherry",
        "severity": "None",
        "description": "Your cherry tree appears healthy with no visible disease symptoms!",
        "symptoms": [
            "Green, healthy foliage",
            "No disease symptoms visible",
            "Normal growth and fruit development",
            "Good tree vigor"
        ],
        "causes": ["Plant is healthy"],
        "treatment": ["Continue current care routine"],
        "prevention": [
            "Maintain proper watering",
            "Regular pruning for structure",
            "Monitor for pests and diseases",
            "Ensure adequate nutrition",
            "Good orchard sanitation"
        ],
        "organic_treatment": [
            "Continue organic practices",
            "Maintain soil health",
            "Use beneficial insects",
            "Regular monitoring"
        ]
    },
    
    # CORN DISEASES
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "disease_name": "Corn Gray Leaf Spot",
        "scientific_name": "Cercospora zeae-maydis",
        "crop": "Corn",
        "severity": "High",
        "description": "A fungal disease causing rectangular gray lesions on corn leaves.",
        "symptoms": [
            "Rectangular gray to tan lesions",
            "Lesions parallel to leaf veins",
            "Yellow halos around spots",
            "Premature leaf death",
            "Reduced photosynthesis"
        ],
        "causes": [
            "Cercospora zeae-maydis fungus",
            "Warm, humid weather",
            "High relative humidity",
            "Infected crop residue"
        ],
        "treatment": [
            "Apply fungicides at first symptoms",
            "Use resistant corn hybrids",
            "Improve field drainage",
            "Remove infected plant debris"
        ],
        "prevention": [
            "Crop rotation with non-host crops",
            "Use resistant varieties",
            "Proper field sanitation",
            "Balanced fertilization"
        ],
        "organic_treatment": [
            "Biological control agents",
            "Crop rotation",
            "Organic fungicide applications",
            "Soil health improvement"
        ]
    },
    
    "Corn_(maize)___Common_rust_": {
        "disease_name": "Corn Common Rust",
        "scientific_name": "Puccinia sorghi",
        "crop": "Corn",
        "severity": "Moderate",
        "description": "A fungal disease causing reddish-brown pustules on corn leaves.",
        "symptoms": [
            "Small, oval reddish-brown pustules",
            "Pustules on both leaf surfaces",
            "Leaves may turn yellow",
            "Premature leaf death",
            "Reduced plant vigor"
        ],
        "causes": [
            "Puccinia sorghi fungus",
            "Cool, moist weather",
            "High humidity",
            "Wind-dispersed spores"
        ],
        "treatment": [
            "Apply fungicides if severe",
            "Use resistant corn varieties",
            "Monitor weather conditions",
            "Remove infected plant material"
        ],
        "prevention": [
            "Plant resistant hybrids",
            "Proper field sanitation",
            "Monitor environmental conditions",
            "Balanced plant nutrition"
        ],
        "organic_treatment": [
            "Resistant variety selection",
            "Cultural control methods",
            "Organic fungicides",
            "Field sanitation"
        ]
    },
    
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
    },
    
    "Corn_(maize)___healthy": {
        "disease_name": "Healthy Corn",
        "scientific_name": "N/A",
        "crop": "Corn",
        "severity": "None",
        "description": "Your corn plant appears healthy with no visible disease symptoms!",
        "symptoms": [
            "Green, healthy leaves",
            "No disease symptoms visible",
            "Normal growth and development",
            "Good plant vigor"
        ],
        "causes": ["Plant is healthy"],
        "treatment": ["Continue current management practices"],
        "prevention": [
            "Maintain proper fertilization",
            "Ensure adequate water supply",
            "Monitor for pests and diseases",
            "Practice crop rotation",
            "Use quality seeds"
        ],
        "organic_treatment": [
            "Continue organic practices",
            "Maintain soil health",
            "Integrated pest management",
            "Regular monitoring"
        ]
    },
    
    # GRAPE DISEASES
    "Grape___Black_rot": {
        "disease_name": "Grape Black Rot",
        "scientific_name": "Guignardia bidwellii",
        "crop": "Grape",
        "severity": "High",
        "description": "A serious fungal disease that affects grape clusters and leaves, causing black, mummified berries.",
        "symptoms": [
            "Small, circular brown spots on leaves with light centers",
            "Black, shriveled berries (mummies)",
            "Reddish-brown lesions on shoots and tendrils",
            "Premature leaf drop",
            "Reduced grape quality and yield"
        ],
        "causes": [
            "Fungal infection by Guignardia bidwellii",
            "Warm, humid weather conditions",
            "Poor air circulation in vineyard",
            "Infected plant debris from previous season"
        ],
        "treatment": [
            "Remove and destroy infected plant material immediately",
            "Apply copper-based fungicides during growing season",
            "Use systemic fungicides like myclobutanil or tebuconazole",
            "Improve air circulation around vines",
            "Prune infected shoots and clusters"
        ],
        "prevention": [
            "Plant resistant grape varieties when possible",
            "Ensure excellent air circulation in vineyard",
            "Avoid overhead watering systems",
            "Regular pruning for proper vine structure",
            "Clean up and destroy fallen leaves and mummified berries"
        ],
        "organic_treatment": [
            "Bordeaux mixture (copper sulfate + lime)",
            "Sulfur-based fungicides",
            "Proper sanitation and cultural practices",
            "Beneficial microorganism applications"
        ]
    },
    
    "Grape___Esca_(Black_Measles)": {
        "disease_name": "Grape Esca (Black Measles)",
        "scientific_name": "Phaeomoniella chlamydospora",
        "crop": "Grape",
        "severity": "Very High",
        "description": "A complex trunk disease affecting mature grapevines, often leading to vine death.",
        "symptoms": [
            "Tiger-stripe yellowing pattern on leaves",
            "Black spots on grape berries",
            "Sudden wilting of shoots (apoplexy)",
            "Wood discoloration in trunk",
            "Dieback of cordons and arms"
        ],
        "causes": [
            "Complex of fungal pathogens",
            "Pruning wounds as entry points",
            "Stress conditions",
            "Age of vines (more common in older vines)"
        ],
        "treatment": [
            "Remove infected wood immediately",
            "Apply wound protectants after pruning",
            "Trunk injection treatments (consult specialist)",
            "Improve vine nutrition and water management",
            "Consider vine replacement in severe cases"
        ],
        "prevention": [
            "Use proper pruning techniques and timing",
            "Apply wound protectants immediately after pruning",
            "Maintain vine health and reduce stress",
            "Use clean, sterilized pruning tools",
            "Plant certified disease-free vines"
        ],
        "organic_treatment": [
            "Proper pruning and sanitation",
            "Wound protection with natural compounds",
            "Stress reduction through proper care",
            "Biological control agents (research ongoing)"
        ]
    },
    
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "disease_name": "Grape Leaf Blight (Isariopsis Leaf Spot)",
        "scientific_name": "Isariopsis clavispora",
        "crop": "Grape",
        "severity": "Moderate",
        "description": "A fungal disease causing leaf spots and premature defoliation in grapevines.",
        "symptoms": [
            "Small, dark brown to black spots on leaves",
            "Yellow halos around leaf spots",
            "Premature yellowing and dropping of leaves",
            "Reduced photosynthetic capacity",
            "Weakened vine vigor"
        ],
        "causes": [
            "Isariopsis clavispora fungal infection",
            "High humidity and moisture",
            "Poor air circulation",
            "Overhead irrigation"
        ],
        "treatment": [
            "Apply copper-based fungicides",
            "Remove and destroy infected leaves",
            "Improve air circulation through pruning",
            "Use preventive fungicide sprays",
            "Adjust irrigation to avoid leaf wetness"
        ],
        "prevention": [
            "Avoid overhead irrigation systems",
            "Ensure proper vine spacing for air circulation",
            "Regular monitoring for early detection",
            "Timely fungicide applications during susceptible periods",
            "Maintain good vineyard sanitation"
        ],
        "organic_treatment": [
            "Copper sulfate applications",
            "Sulfur-based fungicides",
            "Cultural control methods",
            "Proper vine training and pruning"
        ]
    },
    
    "Grape___healthy": {
        "disease_name": "Healthy Grape",
        "scientific_name": "N/A",
        "crop": "Grape",
        "severity": "None",
        "description": "Your grape vine appears healthy with vibrant foliage and no disease symptoms!",
        "symptoms": [
            "Green, healthy leaves",
            "No visible disease symptoms",
            "Normal growth and berry development",
            "Good vine vigor"
        ],
        "causes": ["Plant is healthy"],
        "treatment": ["Continue current vineyard management practices"],
        "prevention": [
            "Maintain proper pruning schedule",
            "Ensure adequate nutrition",
            "Monitor regularly for pests and diseases",
            "Maintain good air circulation",
            "Practice proper irrigation management"
        ],
        "organic_treatment": [
            "Continue organic viticulture practices",
            "Regular soil health monitoring",
            "Integrated pest management",
            "Natural disease prevention methods"
        ]
    },
    
    # ORANGE DISEASES
    "Orange___Haunglongbing_(Citrus_greening)": {
        "disease_name": "Citrus Greening (HLB)",
        "scientific_name": "Candidatus Liberibacter asiaticus",
        "crop": "Orange",
        "severity": "Very High",
        "description": "A devastating bacterial disease transmitted by Asian citrus psyllid, causing tree decline and death.",
        "symptoms": [
            "Yellow shoots and branches",
            "Mottled, yellowing leaves",
            "Small, lopsided fruit",
            "Bitter, unusable fruit",
            "Tree decline and death"
        ],
        "causes": [
            "Candidatus Liberibacter asiaticus bacteria",
            "Asian citrus psyllid transmission",
            "Infected plant material",
            "Poor tree health"
        ],
        "treatment": [
            "Remove infected trees immediately",
            "Control psyllid populations",
            "Apply systemic antibiotics (limited effectiveness)",
            "Improve tree nutrition",
            "No cure available - prevention is key"
        ],
        "prevention": [
            "Control Asian citrus psyllid",
            "Use certified disease-free nursery stock",
            "Regular monitoring and early detection",
            "Quarantine measures",
            "Remove infected trees promptly"
        ],
        "organic_treatment": [
            "Beneficial insect release",
            "Physical removal of infected trees",
            "Psyllid monitoring and control",
            "Tree health maintenance"
        ]
    },
    
    # PEACH DISEASES
    "Peach___Bacterial_spot": {
        "disease_name": "Peach Bacterial Spot",
        "scientific_name": "Xanthomonas arboricola",
        "crop": "Peach",
        "severity": "High",
        "description": "A bacterial disease causing spots on leaves and fruit of peach trees.",
        "symptoms": [
            "Small, dark spots on leaves",
            "Spots have yellow halos",
            "Fruit develops raised, rough spots",
            "Premature leaf drop",
            "Reduced fruit quality"
        ],
        "causes": [
            "Xanthomonas arboricola bacteria",
            "Warm, wet weather",
            "Overhead irrigation",
            "Wounds from insects or hail",
            "Contaminated pruning tools"
        ],
        "treatment": [
            "Apply copper-based bactericides",
            "Remove infected plant material",
            "Improve air circulation",
            "Use drip irrigation",
            "Apply preventive sprays"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Avoid overhead watering",
            "Proper pruning for air circulation",
            "Disinfect tools between trees",
            "Remove fallen leaves and debris"
        ],
        "organic_treatment": [
            "Copper sulfate applications",
            "Proper sanitation practices",
            "Cultural control methods",
            "Beneficial bacteria applications"
        ]
    },
    
    "Peach___healthy": {
        "disease_name": "Healthy Peach",
        "scientific_name": "N/A",
        "crop": "Peach",
        "severity": "None",
        "description": "Your peach tree appears healthy with no visible disease symptoms!",
        "symptoms": [
            "Green, healthy foliage",
            "No disease symptoms visible",
            "Normal growth and fruit development",
            "Good tree vigor"
        ],
        "causes": ["Plant is healthy"],
        "treatment": ["Continue current care routine"],
        "prevention": [
            "Maintain proper watering",
            "Regular pruning for structure",
            "Monitor for pests and diseases",
            "Ensure adequate nutrition",
            "Good orchard sanitation"
        ],
        "organic_treatment": [
            "Continue organic practices",
            "Maintain soil health",
            "Use beneficial insects",
            "Regular monitoring"
        ]
    },
    
    # PEPPER DISEASES
    "Pepper,_bell___Bacterial_spot": {
        "disease_name": "Bell Pepper Bacterial Spot",
        "scientific_name": "Xanthomonas euvesicatoria",
        "crop": "Bell Pepper",
        "severity": "High",
        "description": "A bacterial disease causing spots on leaves, stems, and fruit of bell pepper plants.",
        "symptoms": [
            "Small, dark brown spots on leaves",
            "Spots have yellow halos",
            "Raised, scabby spots on fruit",
            "Premature leaf drop",
            "Reduced fruit quality and yield"
        ],
        "causes": [
            "Xanthomonas euvesicatoria bacteria",
            "Warm, humid weather",
            "Overhead watering",
            "Contaminated seeds or transplants",
            "Wounds from insects or tools"
        ],
        "treatment": [
            "Apply copper-based bactericides",
            "Remove infected plant material",
            "Improve air circulation",
            "Use drip irrigation",
            "Apply preventive sprays"
        ],
        "prevention": [
            "Use certified disease-free seeds",
            "Avoid overhead watering",
            "Rotate crops annually",
            "Disinfect tools between plants",
            "Plant resistant varieties"
        ],
        "organic_treatment": [
            "Copper sulfate applications",
            "Proper sanitation practices",
            "Beneficial bacteria applications",
            "Cultural control methods"
        ]
    },
    
    "Pepper,_bell___healthy": {
        "disease_name": "Healthy Bell Pepper",
        "scientific_name": "N/A",
        "crop": "Bell Pepper",
        "severity": "None",
        "description": "Your bell pepper plant appears healthy with no visible disease symptoms!",
        "symptoms": [
            "Green, healthy foliage",
            "No disease symptoms visible",
            "Normal growth and fruit development",
            "Good plant vigor"
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
    
    # POTATO DISEASES
    "Potato___Early_blight": {
        "disease_name": "Potato Early Blight",
        "scientific_name": "Alternaria solani",
        "crop": "Potato",
        "severity": "Moderate",
        "description": "A common fungal disease affecting potato plants, especially older leaves.",
        "symptoms": [
            "Brown spots with concentric rings on leaves",
            "Yellowing and dropping of lower leaves",
            "Dark, sunken spots on tubers",
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
    
    "Potato___Late_blight": {
        "disease_name": "Potato Late Blight",
        "scientific_name": "Phytophthora infestans",
        "crop": "Potato",
        "severity": "Very High",
        "description": "A devastating disease that can destroy entire potato crops quickly in favorable conditions.",
        "symptoms": [
            "Dark, water-soaked spots on leaves",
            "White mold growth on leaf undersides",
            "Brown, firm rot on tubers",
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
    
    "Potato___healthy": {
        "disease_name": "Healthy Potato",
        "scientific_name": "N/A",
        "crop": "Potato",
        "severity": "None",
        "description": "Your potato plant appears healthy with no visible disease symptoms!",
        "symptoms": [
            "Green, healthy foliage",
            "No disease symptoms visible",
            "Normal growth and tuber development",
            "Good plant vigor"
        ],
        "causes": ["Plant is healthy"],
        "treatment": ["Continue current care routine"],
        "prevention": [
            "Maintain consistent watering",
            "Ensure proper hilling",
            "Monitor regularly for pests and diseases",
            "Ensure proper nutrition",
            "Practice crop rotation"
        ],
        "organic_treatment": [
            "Continue organic practices",
            "Maintain soil health",
            "Use beneficial microorganisms",
            "Regular monitoring"
        ]
    },
    
    # RASPBERRY DISEASES
    "Raspberry___healthy": {
        "disease_name": "Healthy Raspberry",
        "scientific_name": "N/A",
        "crop": "Raspberry",
        "severity": "None",
        "description": "Your raspberry plant appears healthy with no visible disease symptoms!",
        "symptoms": [
            "Green, healthy canes and leaves",
            "No disease symptoms visible",
            "Normal growth and fruit development",
            "Good plant vigor"
        ],
        "causes": ["Plant is healthy"],
        "treatment": ["Continue current care routine"],
        "prevention": [
            "Maintain proper pruning",
            "Ensure good air circulation",
            "Monitor regularly for pests and diseases",
            "Ensure proper nutrition",
            "Remove old canes annually"
        ],
        "organic_treatment": [
            "Continue organic practices",
            "Maintain soil health",
            "Use beneficial insects",
            "Regular monitoring"
        ]
    },
    
    # SOYBEAN DISEASES
    "Soybean___healthy": {
        "disease_name": "Healthy Soybean",
        "scientific_name": "N/A",
        "crop": "Soybean",
        "severity": "None",
        "description": "Your soybean plant appears healthy with no visible disease symptoms!",
        "symptoms": [
            "Green, healthy foliage",
            "No disease symptoms visible",
            "Normal growth and pod development",
            "Good plant vigor"
        ],
        "causes": ["Plant is healthy"],
        "treatment": ["Continue current management practices"],
        "prevention": [
            "Maintain proper fertilization",
            "Ensure adequate water supply",
            "Monitor for pests and diseases",
            "Practice crop rotation",
            "Use quality seeds"
        ],
        "organic_treatment": [
            "Continue organic practices",
            "Maintain soil health",
            "Integrated pest management",
            "Regular monitoring"
        ]
    },
    
    # SQUASH DISEASES
    "Squash___Powdery_mildew": {
        "disease_name": "Squash Powdery Mildew",
        "scientific_name": "Podosphaera xanthii",
        "crop": "Squash",
        "severity": "Moderate",
        "description": "A fungal disease causing white powdery growth on squash leaves and stems.",
        "symptoms": [
            "White powdery coating on leaves",
            "Yellowing and browning of leaves",
            "Stunted plant growth",
            "Reduced fruit quality",
            "Premature leaf drop"
        ],
        "causes": [
            "Podosphaera xanthii fungus",
            "High humidity with dry conditions",
            "Poor air circulation",
            "Overcrowded plantings"
        ],
        "treatment": [
            "Apply sulfur-based fungicides",
            "Use systemic fungicides",
            "Improve air circulation",
            "Remove infected plant material",
            "Apply horticultural oils"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Ensure proper spacing",
            "Improve air circulation",
            "Avoid overhead watering",
            "Regular monitoring"
        ],
        "organic_treatment": [
            "Sulfur dust or spray",
            "Baking soda solution",
            "Neem oil application",
            "Proper plant spacing"
        ]
    },
    
    # STRAWBERRY DISEASES
    "Strawberry___Leaf_scorch": {
        "disease_name": "Strawberry Leaf Scorch",
        "scientific_name": "Diplocarpon earlianum",
        "crop": "Strawberry",
        "severity": "Moderate",
        "description": "A fungal disease causing leaf spots and scorching on strawberry plants.",
        "symptoms": [
            "Small, dark purple spots on leaves",
            "Spots develop gray centers",
            "Leaves appear scorched or burned",
            "Premature leaf drop",
            "Reduced plant vigor"
        ],
        "causes": [
            "Diplocarpon earlianum fungus",
            "Warm, humid weather",
            "Poor air circulation",
            "Overhead watering",
            "Infected plant debris"
        ],
        "treatment": [
            "Apply fungicides containing captan",
            "Remove infected leaves",
            "Improve air circulation",
            "Use drip irrigation",
            "Apply preventive sprays"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Ensure proper spacing",
            "Avoid overhead watering",
            "Remove plant debris",
            "Regular monitoring"
        ],
        "organic_treatment": [
            "Copper-based fungicides",
            "Proper sanitation",
            "Cultural control methods",
            "Beneficial microorganisms"
        ]
    },
    
    "Strawberry___healthy": {
        "disease_name": "Healthy Strawberry",
        "scientific_name": "N/A",
        "crop": "Strawberry",
        "severity": "None",
        "description": "Your strawberry plant appears healthy with no visible disease symptoms!",
        "symptoms": [
            "Green, healthy foliage",
            "No disease symptoms visible",
            "Normal growth and fruit development",
            "Good plant vigor"
        ],
        "causes": ["Plant is healthy"],
        "treatment": ["Continue current care routine"],
        "prevention": [
            "Maintain proper watering",
            "Ensure good drainage",
            "Monitor regularly for pests and diseases",
            "Ensure proper nutrition",
            "Remove old leaves regularly"
        ],
        "organic_treatment": [
            "Continue organic practices",
            "Maintain soil health",
            "Use beneficial insects",
            "Regular monitoring"
        ]
    },
    
    # TOMATO DISEASES - COMPLETE SET
    "Tomato___Bacterial_spot": {
        "disease_name": "Tomato Bacterial Spot",
        "scientific_name": "Xanthomonas vesicatoria",
        "crop": "Tomato",
        "severity": "High",
        "description": "A bacterial disease causing spots on leaves, stems, and fruit of tomato plants.",
        "symptoms": [
            "Small, dark brown spots on leaves",
            "Spots have yellow halos",
            "Fruit develops raised, scabby spots",
            "Premature leaf drop",
            "Reduced fruit quality"
        ],
        "causes": [
            "Xanthomonas bacterial infection",
            "Warm, humid weather",
            "Overhead watering",
            "Contaminated seeds or transplants",
            "Wounds from insects or tools"
        ],
        "treatment": [
            "Apply copper-based bactericides",
            "Remove infected plant material",
            "Improve air circulation",
            "Use drip irrigation",
            "Apply preventive sprays"
        ],
        "prevention": [
            "Use certified disease-free seeds",
            "Avoid overhead watering",
            "Rotate crops annually",
            "Disinfect tools between plants",
            "Plant resistant varieties"
        ],
        "organic_treatment": [
            "Copper sulfate applications",
            "Proper sanitation practices",
            "Beneficial bacteria applications",
            "Cultural control methods"
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
    
    "Tomato___Leaf_Mold": {
        "disease_name": "Tomato Leaf Mold",
        "scientific_name": "Passalora fulva",
        "crop": "Tomato",
        "severity": "Moderate",
        "description": "A fungal disease that primarily affects greenhouse tomatoes, causing yellowing and moldy growth on leaves.",
        "symptoms": [
            "Yellow spots on upper leaf surface",
            "Olive-green to brown mold on leaf undersides",
            "Leaves turn brown and wither",
            "Reduced photosynthesis",
            "Premature defoliation"
        ],
        "causes": [
            "Passalora fulva fungal infection",
            "High humidity (above 85%)",
            "Poor air circulation",
            "Temperature between 72-75°F",
            "Greenhouse conditions"
        ],
        "treatment": [
            "Improve ventilation and air circulation",
            "Reduce humidity levels below 85%",
            "Apply copper-based fungicides",
            "Remove infected leaves immediately",
            "Use resistant tomato varieties"
        ],
        "prevention": [
            "Maintain good air circulation",
            "Control greenhouse humidity",
            "Space plants properly",
            "Use drip irrigation instead of overhead watering",
            "Regular monitoring for early detection"
        ],
        "organic_treatment": [
            "Improve air circulation naturally",
            "Use biological fungicides",
            "Apply compost tea foliar spray",
            "Remove affected plant material"
        ]
    },
    
    "Tomato___Septoria_leaf_spot": {
        "disease_name": "Tomato Septoria Leaf Spot",
        "scientific_name": "Septoria lycopersici",
        "crop": "Tomato",
        "severity": "Moderate",
        "description": "A fungal disease causing distinctive spotted lesions on tomato leaves.",
        "symptoms": [
            "Small, circular spots with gray centers",
            "Dark brown borders around spots",
            "Black specks (pycnidia) in spot centers",
            "Lower leaves affected first",
            "Progressive upward spread"
        ],
        "causes": [
            "Septoria lycopersici fungus",
            "Warm, wet weather",
            "High humidity",
            "Poor air circulation",
            "Infected plant debris"
        ],
        "treatment": [
            "Apply fungicides containing chlorothalonil",
            "Remove lower infected leaves",
            "Improve air circulation",
            "Mulch around plants",
            "Water at soil level only"
        ],
        "prevention": [
            "Use resistant varieties",
            "Proper plant spacing",
            "Avoid overhead irrigation",
            "Remove plant debris",
            "Crop rotation"
        ],
        "organic_treatment": [
            "Neem oil applications",
            "Baking soda spray",
            "Proper mulching",
            "Cultural practices"
        ]
    },
    
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "disease_name": "Tomato Spider Mites (Two-spotted)",
        "scientific_name": "Tetranychus urticae",
        "crop": "Tomato",
        "severity": "Moderate",
        "description": "Tiny spider mites that feed on tomato leaves, causing stippling and webbing.",
        "symptoms": [
            "Fine stippling on leaf surface",
            "Yellow or bronze discoloration",
            "Fine webbing on leaves",
            "Premature leaf drop",
            "Reduced plant vigor"
        ],
        "causes": [
            "Two-spotted spider mite infestation",
            "Hot, dry conditions",
            "Dusty environments",
            "Overuse of broad-spectrum pesticides",
            "Stressed plants"
        ],
        "treatment": [
            "Apply miticides or insecticidal soap",
            "Increase humidity around plants",
            "Use predatory mites",
            "Remove heavily infested leaves",
            "Spray with water to dislodge mites"
        ],
        "prevention": [
            "Maintain adequate soil moisture",
            "Avoid dusty conditions",
            "Use beneficial insects",
            "Regular monitoring",
            "Proper plant nutrition"
        ],
        "organic_treatment": [
            "Neem oil spray",
            "Insecticidal soap",
            "Predatory mite release",
            "Strong water spray"
        ]
    },
    
    "Tomato___Target_Spot": {
        "disease_name": "Tomato Target Spot",
        "scientific_name": "Corynespora cassiicola",
        "crop": "Tomato",
        "severity": "Moderate",
        "description": "A fungal disease causing target-like spots on tomato leaves and fruit.",
        "symptoms": [
            "Brown spots with concentric rings",
            "Target-like appearance",
            "Spots on leaves, stems, and fruit",
            "Premature defoliation",
            "Reduced fruit quality"
        ],
        "causes": [
            "Corynespora cassiicola fungus",
            "Warm, humid conditions",
            "Poor air circulation",
            "Overhead watering",
            "Plant stress"
        ],
        "treatment": [
            "Apply fungicides with active ingredients like azoxystrobin",
            "Remove infected plant material",
            "Improve air circulation",
            "Use drip irrigation",
            "Maintain plant health"
        ],
        "prevention": [
            "Use resistant varieties",
            "Proper plant spacing",
            "Avoid overhead watering",
            "Crop rotation",
            "Good sanitation practices"
        ],
        "organic_treatment": [
            "Copper-based fungicides",
            "Biological control agents",
            "Cultural practices",
            "Proper plant nutrition"
        ]
    },
    
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "disease_name": "Tomato Yellow Leaf Curl Virus",
        "scientific_name": "TYLCV",
        "crop": "Tomato",
        "severity": "Very High",
        "description": "A viral disease transmitted by whiteflies, causing severe stunting and yield loss.",
        "symptoms": [
            "Upward curling of leaves",
            "Yellow leaf margins",
            "Stunted plant growth",
            "Reduced fruit size and number",
            "Interveinal yellowing"
        ],
        "causes": [
            "Tomato Yellow Leaf Curl Virus",
            "Whitefly transmission",
            "Infected transplants",
            "Nearby infected plants",
            "Warm weather conditions"
        ],
        "treatment": [
            "Remove infected plants immediately",
            "Control whitefly populations",
            "Use reflective mulches",
            "Apply insecticides for whitefly control",
            "No cure available - prevention is key"
        ],
        "prevention": [
            "Use virus-resistant varieties",
            "Control whitefly populations",
            "Use physical barriers (screens)",
            "Remove infected plants promptly",
            "Monitor for early symptoms"
        ],
        "organic_treatment": [
            "Beneficial insect release",
            "Reflective mulches",
            "Physical removal of infected plants",
            "Whitefly traps"
        ]
    },
    
    "Tomato___Tomato_mosaic_virus": {
        "disease_name": "Tomato Mosaic Virus",
        "scientific_name": "ToMV",
        "crop": "Tomato",
        "severity": "High",
        "description": "A viral disease causing mosaic patterns on leaves and reduced fruit quality.",
        "symptoms": [
            "Mosaic pattern on leaves",
            "Light and dark green patches",
            "Leaf distortion",
            "Stunted growth",
            "Reduced fruit quality"
        ],
        "causes": [
            "Tomato Mosaic Virus",
            "Mechanical transmission",
            "Contaminated tools",
            "Infected seeds",
            "Human handling"
        ],
        "treatment": [
            "Remove infected plants",
            "Disinfect tools and hands",
            "No chemical treatment available",
            "Focus on prevention",
            "Control aphid vectors"
        ],
        "prevention": [
            "Use certified virus-free seeds",
            "Disinfect tools regularly",
            "Wash hands between plants",
            "Control aphid populations",
            "Remove infected plants promptly"
        ],
        "organic_treatment": [
            "Sanitation practices",
            "Physical removal",
            "Tool disinfection",
            "Beneficial insect management"
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
