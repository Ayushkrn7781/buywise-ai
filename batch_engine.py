"""
Batch analysis engine - analyze multiple reviews for a product.
"""

import re
import os
import spacy
from collections import Counter
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from product_identifier import identify_product

# ==================== LOAD MODELS ====================

nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()

TRAINED_MODEL_PATH = "./trained_sentiment_model"

if os.path.exists(TRAINED_MODEL_PATH):
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model=TRAINED_MODEL_PATH,
        tokenizer=TRAINED_MODEL_PATH
    )
    USING_CUSTOM_MODEL = True
else:
    sentiment_pipeline = pipeline("sentiment-analysis")
    USING_CUSTOM_MODEL = False

# Aspect keywords
ASPECT_KEYWORDS = {
    "battery life": ["battery", "batteries", "charge", "charging", "battery life", "battery backup", "hours"],
    "sound quality": ["sound", "audio", "bass", "treble", "sound quality", "audio quality"],
    "build quality": ["build", "build quality", "construction", "sturdy", "flimsy", "plastic", "crack", "hinge"],
    "comfort": ["comfortable", "comfort", "ear cushion", "clamping", "fit", "wear"],
    "connectivity": ["bluetooth", "connection", "disconnect", "wireless", "pairing", "connectivity"],
    "microphone quality": ["microphone", "mic", "call quality", "voice", "calling"],
    "noise cancellation": ["noise cancellation", "anc", "noise canceling", "blocking noise"],
    "software": ["app", "software", "firmware", "eq", "settings"],
    "value for money": ["value", "worth", "price", "expensive", "cheap", "overpriced"],
    "design": ["design", "look", "looks", "aesthetic", "appearance", "stylish"],
    "ease of use": ["easy to use", "user friendly", "controls", "touch controls", "buttons"],
    "performance": ["performance", "fast", "slow", "lag", "responsive", "snappy"],
    "display": ["display", "screen", "brightness", "resolution"],
    "camera": ["camera", "photos", "pictures", "video quality"],
    "customer service": ["customer service", "support", "help", "warranty"],
    "durability": ["durable", "durability", "last", "broke", "broken", "damage"],
    "packaging": ["packaging", "box", "unboxing"],
    "delivery": ["delivery", "shipping", "arrived"],
    "size": ["size", "compact", "bulky", "small", "large"],
    "weight": ["weight", "heavy", "light", "lightweight"],
    "material quality": ["material", "materials", "leather", "metal", "fabric"],
    "reliability": ["reliable", "reliability", "consistent", "issues", "problems"],
}


# ==================== SINGLE REVIEW ANALYSIS ====================

def analyze_single_review(text):
    """Quick analysis of a single review"""
    text_lower = text.lower()
    
    # VADER sentiment
    scores = sentiment_analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        vader_sentiment = "positive"
    elif scores['compound'] <= -0.05:
        vader_sentiment = "negative"
    else:
        vader_sentiment = "neutral"
    
    # Custom model sentiment
    try:
        result = sentiment_pipeline(text, truncation=True, max_length=512)[0]
        label = result['label'].lower()
        if label in ['positive', 'pos', 'label_2', '2']:
            custom_sentiment = "positive"
        elif label in ['negative', 'neg', 'label_0', '0']:
            custom_sentiment = "negative"
        else:
            custom_sentiment = "neutral"
        custom_confidence = result['score']
    except:
        custom_sentiment = vader_sentiment
        custom_confidence = 0.5
    
    # Find mentioned aspects
    mentioned_aspects = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                mentioned_aspects.append(aspect)
                break
    
    # Get aspect sentiments
    aspect_sentiments = {}
    sentences = [sent.text for sent in nlp(text).sents]
    
    for aspect in mentioned_aspects:
        keywords = ASPECT_KEYWORDS[aspect]
        relevant_sents = []
        for sent in sentences:
            sent_lower = sent.lower()
            for kw in keywords:
                if kw in sent_lower:
                    relevant_sents.append(sent)
                    break
        
        if relevant_sents:
            sent_sentiments = []
            for sent in relevant_sents:
                s = sentiment_analyzer.polarity_scores(sent)
                if s['compound'] >= 0.05:
                    sent_sentiments.append("positive")
                elif s['compound'] <= -0.05:
                    sent_sentiments.append("negative")
                else:
                    sent_sentiments.append("neutral")
            
            pos = sent_sentiments.count("positive")
            neg = sent_sentiments.count("negative")
            neu = sent_sentiments.count("neutral")
            
            if pos > neg and pos > neu:
                asp_sent = "positive"
            elif neg > pos and neg > neu:
                asp_sent = "negative"
            else:
                asp_sent = "neutral"
            
            aspect_sentiments[aspect] = asp_sent
    
    return {
        "vader_sentiment": vader_sentiment,
        "custom_sentiment": custom_sentiment,
        "custom_confidence": custom_confidence,
        "aspects": aspect_sentiments,
        "length": len(text)
    }


# ==================== BATCH ANALYSIS ====================

def batch_analyze(reviews, product_name=None):
    """
    Analyze multiple reviews for a product.
    
    Args:
        reviews: List of review texts
        product_name: Optional product name
    
    Returns:
        Complete batch analysis results
    """
    
    # Step 1: Identify the product
    product_info = identify_product(reviews, product_name)
    
    # Step 2: Analyze each review
    all_results = []
    errors = 0
    
    for i, review in enumerate(reviews):
        if not isinstance(review, str) or len(review.strip()) < 10:
            errors += 1
            continue
        try:
            result = analyze_single_review(review)
            all_results.append(result)
        except Exception as e:
            errors += 1
            continue
    
    if not all_results:
        return {"error": "No valid reviews to analyze"}
    
    # Step 3: Aggregate sentiment
    vader_sentiments = [r["vader_sentiment"] for r in all_results]
    custom_sentiments = [r["custom_sentiment"] for r in all_results]
    
    vader_counts = Counter(vader_sentiments)
    custom_counts = Counter(custom_sentiments)
    total = len(all_results)
    
    # Step 4: Aggregate aspects
    aspect_aggregate = {}
    
    for result in all_results:
        for aspect, sentiment in result["aspects"].items():
            if aspect not in aspect_aggregate:
                aspect_aggregate[aspect] = {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "total": 0
                }
            aspect_aggregate[aspect][sentiment] += 1
            aspect_aggregate[aspect]["total"] += 1
    
    # Calculate percentages and sort
    aspect_summary = []
    for aspect, counts in aspect_aggregate.items():
        t = counts["total"]
        pos_pct = round(counts["positive"] / t * 100, 1)
        neg_pct = round(counts["negative"] / t * 100, 1)
        neu_pct = round(counts["neutral"] / t * 100, 1)
        
        if pos_pct > neg_pct and pos_pct > neu_pct:
            overall = "positive"
        elif neg_pct > pos_pct and neg_pct > neu_pct:
            overall = "negative"
        else:
            overall = "neutral"
        
        aspect_summary.append({
            "aspect": aspect,
            "mentions": t,
            "positive": counts["positive"],
            "negative": counts["negative"],
            "neutral": counts["neutral"],
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "neutral_pct": neu_pct,
            "overall": overall,
            "mention_pct": round(t / total * 100, 1)
        })
    
    # Sort by mentions
    aspect_summary.sort(key=lambda x: x["mentions"], reverse=True)
    
    # Step 5: Determine pros and cons
    pros = [a["aspect"] for a in aspect_summary if a["overall"] == "positive"]
    cons = [a["aspect"] for a in aspect_summary if a["overall"] == "negative"]
    
    # Step 6: Generate verdict
    overall_positive = custom_counts.get("positive", 0) / total * 100
    overall_negative = custom_counts.get("negative", 0) / total * 100
    
    if overall_positive >= 70:
        verdict = "Strongly Recommended"
        verdict_emoji = "🟢"
    elif overall_positive >= 50:
        verdict = "Recommended with Reservations"
        verdict_emoji = "🟡"
    elif overall_negative >= 70:
        verdict = "Not Recommended"
        verdict_emoji = "🔴"
    elif overall_negative >= 50:
        verdict = "Avoid"
        verdict_emoji = "🔴"
    else:
        verdict = "Mixed Reviews"
        verdict_emoji = "🟠"
    
    # Step 7: Find warning signs
    warnings = []
    for a in aspect_summary:
        if a["negative_pct"] >= 60 and a["mentions"] >= total * 0.1:
            warnings.append(f"{a['aspect']} has {a['negative_pct']}% negative reviews")
    
    return {
        "product": product_info,
        "analysis_summary": {
            "total_reviews": total,
            "skipped_reviews": errors,
            "model_used": "Custom Fine-tuned" if USING_CUSTOM_MODEL else "Pre-trained"
        },
        "sentiment_distribution": {
            "vader": {
                "positive": vader_counts.get("positive", 0),
                "neutral": vader_counts.get("neutral", 0),
                "negative": vader_counts.get("negative", 0),
                "positive_pct": round(vader_counts.get("positive", 0) / total * 100, 1),
                "neutral_pct": round(vader_counts.get("neutral", 0) / total * 100, 1),
                "negative_pct": round(vader_counts.get("negative", 0) / total * 100, 1)
            },
            "custom": {
                "positive": custom_counts.get("positive", 0),
                "neutral": custom_counts.get("neutral", 0),
                "negative": custom_counts.get("negative", 0),
                "positive_pct": round(custom_counts.get("positive", 0) / total * 100, 1),
                "neutral_pct": round(custom_counts.get("neutral", 0) / total * 100, 1),
                "negative_pct": round(custom_counts.get("negative", 0) / total * 100, 1)
            }
        },
        "aspects": aspect_summary,
        "pros": pros,
        "cons": cons,
        "warnings": warnings,
        "verdict": {
            "text": verdict,
            "emoji": verdict_emoji,
            "positive_pct": round(overall_positive, 1),
            "negative_pct": round(overall_negative, 1)
        },
        "individual_results": all_results
    }