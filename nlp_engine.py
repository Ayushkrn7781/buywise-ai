import re
import os
import spacy
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==================== LOAD MODELS ====================

nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()

TRAINED_MODEL_PATH = "./trained_sentiment_model"

if os.path.exists(TRAINED_MODEL_PATH):
    print(f"Loading YOUR trained model from {TRAINED_MODEL_PATH}")
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model=TRAINED_MODEL_PATH,
        tokenizer=TRAINED_MODEL_PATH
    )
    USING_CUSTOM_MODEL = True
else:
    print("WARNING: No trained model found. Using default pre-trained model.")
    sentiment_pipeline = pipeline("sentiment-analysis")
    USING_CUSTOM_MODEL = False

aspect_pipeline = pipeline(
    "zero-shot-classification", 
    model="facebook/bart-large-mnli"
)

# Aspect keywords for better matching
ASPECT_KEYWORDS = {
    "battery life": ["battery", "batteries", "charge", "charging", "battery life", "battery backup", "battery performance", "hours of playback", "battery drain"],
    "sound quality": ["sound", "audio", "bass", "treble", "sound quality", "audio quality", "sound output"],
    "build quality": ["build", "build quality", "construction", "sturdy", "flimsy", "plastic", "crack", "hinge", "durability"],
    "comfort": ["comfortable", "comfort", "ear cushion", "clamping", "ear pressure", "fit", "wear"],
    "connectivity": ["bluetooth", "connection", "connect", "disconnect", "wireless", "pairing", "connectivity"],
    "microphone quality": ["microphone", "mic", "call quality", "voice", "calling"],
    "noise cancellation": ["noise cancellation", "anc", "noise canceling", "blocking noise", "noise isolation"],
    "software": ["app", "software", "firmware", "application", "eq", "settings"],
    "value for money": ["value", "worth", "price", "expensive", "cheap", "overpriced", "budget"],
    "design": ["design", "look", "looks", "aesthetics", "appearance", "stylish"],
    "ease of use": ["easy to use", "user friendly", "controls", "touch controls", "buttons"],
    "performance": ["performance", "fast", "slow", "lag", "responsive", "snappy"],
    "display": ["display", "screen", "brightness", "resolution"],
    "camera": ["camera", "photos", "pictures", "video quality"],
    "customer service": ["customer service", "support", "help", "response", "warranty"],
    "packaging": ["packaging", "box", "unboxing", "accessories included"],
    "delivery": ["delivery", "shipping", "arrived", "arrive", "dispatch"],
    "size": ["size", "compact", "bulky", "small", "large", "big"],
    "weight": ["weight", "heavy", "light", "lightweight"],
    "material quality": ["material", "materials", "leather", "metal", "fabric"],
    "color": ["color", "colour", "finish", "matte", "glossy"],
    "pricing": ["pricing", "price", "cost", "money", "affordable"],
    "reliability": ["reliable", "reliability", "consistent", "issues", "problems"],
    "user interface": ["ui", "interface", "menu", "navigation"]
}


# ==================== PREPROCESSING ====================

def preprocess_review(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==================== SENTIMENT ANALYSIS ====================

def analyze_sentiment_vader(text):
    scores = sentiment_analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        sentiment = "positive"
    elif scores['compound'] <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {
        "sentiment": sentiment,
        "scores": scores,
        "confidence": abs(scores['compound'])
    }


def analyze_sentiment_custom(text):
    result = sentiment_pipeline(
        text,
        truncation=True,
        max_length=512
    )[0]
    
    label = result['label'].lower()
    if label in ['positive', 'pos', 'label_2', '2']:
        label = "positive"
    elif label in ['negative', 'neg', 'label_0', '0']:
        label = "negative"
    else:
        label = "neutral"
    
    return {
        "sentiment": label,
        "confidence": result['score'],
        "raw_label": result['label']
    }


# ==================== ASPECT EXTRACTION (IMPROVED) ====================

def extract_aspects(text):
    """Extract aspects using both zero-shot AND keyword matching"""
    text_lower = text.lower()
    found_aspects = {}
    
    # Method 1: Keyword matching (more reliable for long reviews)
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Count how many times keywords appear
                count = text_lower.count(keyword)
                if aspect not in found_aspects:
                    found_aspects[aspect] = 0
                found_aspects[aspect] += count
    
    # Convert to list with confidence based on mention count
    keyword_aspects = []
    max_count = max(found_aspects.values()) if found_aspects else 1
    
    for aspect, count in found_aspects.items():
        confidence = min(0.95, 0.5 + (count / max_count) * 0.45)
        keyword_aspects.append({
            "aspect": aspect,
            "confidence": round(confidence, 2),
            "mention_count": count
        })
    
    # Method 2: Zero-shot (for aspects keywords might miss)
    try:
        zero_shot_result = aspect_pipeline(text[:500], list(ASPECT_KEYWORDS.keys()), multi_label=True)
        for aspect, score in zip(zero_shot_result['labels'], zero_shot_result['scores']):
            if score > 0.3:  # Higher threshold for zero-shot
                # Check if already found by keywords
                existing = next((a for a in keyword_aspects if a["aspect"] == aspect), None)
                if existing:
                    # Boost confidence if both methods agree
                    existing["confidence"] = min(0.99, round((existing["confidence"] + score) / 2, 2))
                else:
                    keyword_aspects.append({
                        "aspect": aspect,
                        "confidence": round(score, 2),
                        "mention_count": 1
                    })
    except:
        pass
    
    # Sort by confidence
    keyword_aspects.sort(key=lambda x: x["confidence"], reverse=True)
    
    return keyword_aspects


def analyze_aspect_sentiment(text, aspect, keywords=None):
    """Analyze sentiment for a specific aspect using keyword matching"""
    text_lower = text.lower()
    sentences = [sent.text for sent in nlp(text).sents]
    
    # Get keywords for this aspect
    if keywords is None:
        keywords = ASPECT_KEYWORDS.get(aspect, [aspect.lower()])
    
    # Find sentences that mention this aspect
    relevant = []
    for sent in sentences:
        sent_lower = sent.lower()
        for kw in keywords:
            if kw in sent_lower:
                relevant.append(sent)
                break  # Don't add same sentence twice
    
    if not relevant:
        return "neutral", 0.0
    
    # Analyze sentiment of relevant sentences
    sentiments = []
    for sent in relevant:
        scores = sentiment_analyzer.polarity_scores(sent)
        if scores['compound'] >= 0.05:
            sentiments.append("positive")
        elif scores['compound'] <= -0.05:
            sentiments.append("negative")
        else:
            sentiments.append("neutral")
    
    pos = sentiments.count("positive")
    neg = sentiments.count("negative")
    neu = sentiments.count("neutral")
    
    if pos > neg and pos > neu:
        asp_sent = "positive"
    elif neg > pos and neg > neu:
        asp_sent = "negative"
    else:
        asp_sent = "neutral"
    
    total = len(sentiments)
    confidence = max(pos, neg, neu) / total if total > 0 else 0
    return asp_sent, confidence


# ==================== MAIN FUNCTION ====================

def generate_review_intelligence(text):
    cleaned = preprocess_review(text)

    vader_result = analyze_sentiment_vader(cleaned)
    custom_result = analyze_sentiment_custom(cleaned)
    aspects = extract_aspects(cleaned)

    aspect_sentiments = []
    for a in aspects:
        aspect_name = a["aspect"]
        keywords = ASPECT_KEYWORDS.get(aspect_name, [aspect_name.lower()])
        asp_sent, asp_conf = analyze_aspect_sentiment(cleaned, aspect_name, keywords)
        aspect_sentiments.append({
            "aspect": aspect_name,
            "sentiment": asp_sent,
            "aspect_confidence": a["confidence"],
            "sentiment_confidence": asp_conf,
            "mention_count": a.get("mention_count", 1)
        })

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "model_used": "Custom Fine-tuned" if USING_CUSTOM_MODEL else "Pre-trained (default)",
        "overall_sentiment": {
            "vader": vader_result,
            "custom_model": custom_result,
            "combined": custom_result["sentiment"],
            "confidence": (vader_result["confidence"] + custom_result["confidence"]) / 2
        },
        "aspects": aspect_sentiments,
        "pros": [a["aspect"] for a in aspect_sentiments if a["sentiment"] == "positive"],
        "cons": [a["aspect"] for a in aspect_sentiments if a["sentiment"] == "negative"]
    }