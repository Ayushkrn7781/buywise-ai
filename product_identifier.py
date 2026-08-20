"""
Identify what product the reviews are about.
Uses frequency analysis + NER to find the product name.
"""

import re
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

# Common product categories and their indicator words
CATEGORY_INDICATORS = {
    "Headphones/Earphones": ["headphones", "earphones", "earbuds", "headset", "anc", "noise cancellation", "bluetooth headphones", "wireless earbuds"],
    "Laptop": ["laptop", "notebook", "macbook", "thinkpad", "dell", "hp", "lenovo", "screen", "keyboard", "trackpad"],
    "Phone": ["phone", "smartphone", "iphone", "android", "samsung", "pixel", "camera", "battery"],
    "Shoes": ["shoes", "sneakers", "boots", "running shoes", "sole", "fit", "size"],
    "Watch": ["watch", "smartwatch", "fitness tracker", "apple watch", "band", "strap"],
    "Camera": ["camera", "dslr", "lens", "mirrorless", "photo", "video"],
    "Tablet": ["tablet", "ipad", "screen", "stylus"],
    "Speaker": ["speaker", "bluetooth speaker", "sound", "bass", "volume"],
    "TV": ["tv", "television", "smart tv", "screen", "display", "picture"],
    "Clothing": ["shirt", "pants", "jacket", "fabric", "fit", "size", "material"],
}

# Brand patterns
BRAND_PATTERNS = [
    r'\b(sony|samsung|apple|google|amazon|bose|jbl|sennheiser|bose)\b',
    r'\b(dell|hp|lenovo|asus|acer|msi|macbook)\b',
    r'\b(nike|adidas|puma|reebok|new balance)\b',
    r'\b(canon|nikon|sony|fuji|panasonic)\b',
    r'\b(jbl|boAt|realme|oneplus|xiaomi|oppo|vivo)\b',
]

# Model number patterns
MODEL_PATTERNS = [
    r'\b[A-Z]{2,4}[-]?\d{3,5}\b',  # WH-1000XM5, XF-50
    r'\b[A-Z][a-z]+\s+\d{1,2}[a-z]?\b',  # iPhone 15, MacBook Pro
    r'\bGalaxy\s+S\d+\b',  # Galaxy S24
    r'\bAirPods\s*Pro\b',
    r'\bPixel\s*\d+\b',
]


def extract_brands(reviews):
    """Find brands mentioned across reviews"""
    brand_counts = Counter()
    
    for review in reviews:
        for pattern in BRAND_PATTERNS:
            matches = re.findall(pattern, review.lower())
            for match in matches:
                brand_counts[match.title()] += 1
    
    return brand_counts.most_common(5)


def extract_model_numbers(reviews):
    """Find model numbers mentioned across reviews"""
    model_counts = Counter()
    
    for review in reviews:
        for pattern in MODEL_PATTERNS:
            matches = re.findall(pattern, review)
            for match in matches:
                model_counts[match.strip()] += 1
    
    return model_counts.most_common(5)


def extract_product_nouns(reviews):
    """Find most common product-related nouns"""
    noun_counts = Counter()
    
    for review in reviews[:50]:  # Sample to save time
        doc = nlp(review)
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                # Filter out common non-product words
                if token.text.lower() not in ['i', 'it', 'thing', 'time', 'day', 'week', 'month', 'year', 'product', 'item', 'one', 'something', 'anything', 'everything', 'nothing']:
                    noun_counts[token.text.lower()] += 1
    
    return noun_counts.most_common(20)


def detect_category(reviews):
    """Detect product category from indicator words"""
    category_scores = Counter()
    all_text = " ".join(reviews).lower()
    
    for category, indicators in CATEGORY_INDICATORS.items():
        for indicator in indicators:
            count = all_text.count(indicator)
            if count > 0:
                category_scores[category] += count
    
    top_category = category_scores.most_common(1)
    if top_category:
        return top_category[0][0], top_category[0][1]
    return "Unknown", 0


def identify_product(reviews, product_name=None):
    """
    Main function: Identify what product the reviews are about.
    
    Args:
        reviews: List of review texts
        product_name: Optional - if provided, use this instead of detecting
    
    Returns:
        dict with product info
    """
    if product_name and product_name.strip():
        return {
            "name": product_name.strip(),
            "category": "Provided",
            "brands": [],
            "models": [],
            "top_nouns": [],
            "detection_method": "User provided"
        }
    
    # Detect category
    category, category_score = detect_category(reviews)
    
    # Extract brands
    brands = extract_brands(reviews)
    
    # Extract model numbers
    models = extract_model_numbers(reviews)
    
    # Extract common nouns
    top_nouns = extract_product_nouns(reviews)
    
    # Build product name
    name_parts = []
    
    if brands:
        name_parts.append(brands[0][0])
    
    if models:
        name_parts.append(models[0][0])
    
    # If no brand/model found, use category
    if not name_parts:
        name_parts.append(category.split("/")[0])
    
    product_name_detected = " ".join(name_parts) if name_parts else "Unknown Product"
    
    return {
        "name": product_name_detected,
        "category": category,
        "category_score": category_score,
        "brands": brands,
        "models": models,
        "top_nouns": top_nouns[:10],
        "detection_method": "Auto-detected",
        "review_count": len(reviews)
    }