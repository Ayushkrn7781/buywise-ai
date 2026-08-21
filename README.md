# ReviewLens - NLP-Powered Review Intelligence System

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Evaluation](#evaluation)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

ReviewLens is an NLP-based review intelligence system that analyzes product reviews to extract structured insights. Instead of simply classifying a review as positive or negative, the system identifies specific product aspects, determines the sentiment for each aspect, calculates confidence scores, and presents the results in a decision-oriented format.

The system supports two modes of analysis:

- **Single Review Analysis**: Analyze one review at a time with detailed aspect-level sentiment breakdown
- **Batch Analysis**: Process multiple reviews for a product, aggregate results, and generate a product-level verdict

The core sentiment model is fine-tuned on 21,000+ real Amazon reviews, making it domain-specific rather than relying on generic pre-trained models.

---

## Problem Statement

Online marketplaces host thousands of reviews for popular products. Reading all of them is impractical, and simple sentiment scores fail to capture the nuanced opinions expressed in reviews. A customer considering a purchase needs to know:

- What specific aspects do reviewers like or dislike?
- How frequently is each aspect mentioned?
- What is the sentiment distribution for each aspect?
- What is the overall verdict based on aggregated opinions?
- Is the product suitable for their specific use case?

ReviewLens addresses these questions by combining aspect-based sentiment analysis with confidence scoring and aggregate reporting.

---

## Architecture

```
                         INPUT
                           |
              +------------+------------+
              |                         |
         Single Review            Multiple Reviews
              |                         |
      Preprocessing              Preprocessing
              |                         |
      Sentiment Analysis         Sentiment Analysis
      (VADER + Custom)          (VADER + Custom)
              |                         |
      Aspect Extraction          Aspect Extraction
      (Keyword + Zero-shot)     (Keyword + Zero-shot)
              |                         |
      Aspect Sentiment          Aspect Sentiment
              |                         |
         Single Output           Product Identification
                                      |
                               Aggregate Statistics
                                      |
                               Verdict Generation
                                      |
                                 OUTPUT
```

### Component Breakdown

**1. Preprocessing**
- HTML tag removal
- Unicode normalization
- Whitespace normalization
- Text cleaning for consistent analysis

**2. Sentiment Analysis**
- VADER: Rule-based sentiment analysis providing baseline scores
- Custom Fine-tuned Model: DistilBERT fine-tuned on Amazon reviews for domain-specific accuracy
- Combined: Weighted average of both methods

**3. Aspect Extraction**
- Keyword Matching: Searches for predefined aspect-related terms in review text
- Zero-shot Classification: Uses BART-large-mnli to detect aspects not covered by keywords
- Combined: Merges results from both methods with confidence scoring

**4. Aspect Sentiment**
- Sentence-level analysis: Splits review into sentences
- Relevance filtering: Selects only sentences mentioning the target aspect
- Sentiment aggregation: Determines overall aspect sentiment from relevant sentences

**5. Product Identification (Batch Mode)**
- Category detection: Matches review content against product category indicators
- Brand extraction: Uses regex patterns to find brand names
- Model number extraction: Identifies product model numbers
- Noun frequency analysis: Finds common product-related nouns across reviews

---

## Features

### Single Review Mode

| Feature | Description |
|---------|-------------|
| Multi-model Sentiment | Compares VADER, fine-tuned model, and combined results |
| Aspect Extraction | Identifies 200+ product aspects using keyword and zero-shot methods |
| Aspect Sentiment | Determines positive/negative/neutral for each detected aspect |
| Confidence Scores | Provides detection confidence and sentiment confidence for each aspect |
| Pros and Cons | Aggregates aspects into clear pros and cons lists |

### Batch Analysis Mode

| Feature | Description |
|---------|-------------|
| Product Identification | Auto-detects product name, category, brand, and model from reviews |
| Aggregate Sentiment | Calculates positive/neutral/negative distribution across all reviews |
| Aspect Aggregation | Shows mention count, percentage, and sentiment for each aspect |
| Warnings | Flags aspects with high negative sentiment across reviews |
| Verdict | Generates product-level recommendation based on aggregated data |
| CSV Upload | Accepts CSV files with review data for direct analysis |

---

## Project Structure

```
reviewlens/
|
|-- app.py                    # Main Streamlit dashboard (single + batch)
|-- nlp_engine.py             # Single review analysis engine
|-- batch_engine.py           # Batch analysis engine
|-- product_identifier.py     # Product identification from reviews
|-- train_model.py            # Model training script
|-- requirements.txt          # Python dependencies
|-- Amazon_Reviews.csv        # Training dataset
|
|-- trained_sentiment_model/  # Fine-tuned model (generated after training)
|   |-- config.json
|   |-- model.safetensors
|   |-- tokenizer_config.json
|   |-- vocab.txt
|   |-- tokenizer.json
|   |-- training_info.json
|
|-- training_output/          # Training checkpoints (generated during training)
|-- training_logs/            # Training logs (generated during training)
```

### File Responsibilities

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI with sidebar navigation, single review tab, and batch analysis tab |
| `nlp_engine.py` | Loads models, preprocesses text, runs sentiment and aspect analysis for single reviews |
| `batch_engine.py` | Processes multiple reviews, aggregates results, generates verdict |
| `product_identifier.py` | Detects product name, category, brand, and model from review text |
| `train_model.py` | Loads CSV data, preprocesses, fine-tunes DistilBERT, saves model with metrics |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 4 GB free disk space (for models and datasets)
- Internet connection (for first-time model downloads)

### Steps

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv

# 2. Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy language model
python -m spacy download en_core_web_sm
```

### Dependency Installation Notes

If spaCy fails to build from source:
```bash
pip install spacy --only-binary :all:
```

If torchvision warnings appear during Streamlit:
```bash
pip install torchvision
```

---

## Training the Model

### Dataset Requirements

The training script expects a CSV file with at minimum these columns:

| Column | Type | Description |
|--------|------|-------------|
| Review Text | String | The full review text |
| Rating | String or Int | Rating value (e.g., "Rated 1 out of 5 stars" or just 1) |

Optional columns used if present:

| Column | Purpose |
|--------|---------|
| Review Title | Prepended to Review Text for additional context |
| Review Date | Not used in current version |
| Verified Purchase | Not used in current version |

### Rating to Sentiment Mapping

The script maps star ratings to three sentiment classes:

| Rating | Sentiment Label | Numeric Label |
|--------|----------------|---------------|
| 1 star | Negative | 0 |
| 2 stars | Negative | 0 |
| 3 stars | Neutral | 1 |
| 4 stars | Positive | 2 |
| 5 stars | Positive | 2 |

### Training Command

```bash
# Basic training with defaults (3 epochs, batch size 16, 80/20 split)
python train_model.py --data Amazon_Reviews.csv

# Custom parameters
python train_model.py --data Amazon_Reviews.csv --epochs 5 --batch_size 32 --test_size 0.15
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Amazon_Reviews.csv | Path to the CSV file |
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 16 | Training and evaluation batch size |
| `--test_size` | 0.2 | Fraction of data reserved for testing |

### Expected Training Output

```
Loading data from Amazon_Reviews.csv...
Total rows loaded: 21214
Clean dataset size: 21055

Label distribution:
  negative: 14350 (68.2%)
  neutral: 885 (4.2%)
  positive: 5820 (27.6%)

Train size: 16844
Test size: 4211

Loading tokenizer and model: distilbert-base-uncased

==================================================
STARTING TRAINING
==================================================

Epoch 1/3: [============================] 
Epoch 2/3: [============================] 
Epoch 3/3: [============================] 

==================================================
EVALUATION ON TEST SET
==================================================

Accuracy:  0.XXXX
Precision: 0.XXXX
Recall:    0.XXXX
F1 Score:  0.XXXX

Confusion Matrix:
                  Predicted
           Neg   Neu   Pos
Actual NEG  XXXX  XXXX  XXXX
Actual NEU  XXXX  XXXX  XXXX
Actual POS  XXXX  XXXX  XXXX

==================================================
TRAINING COMPLETE!
==================================================
Model saved to: ./trained_sentiment_model
```

### Training Artifacts

After training, the following files are saved in `trained_sentiment_model/`:

- `config.json`: Model architecture configuration
- `model.safetensors`: Fine-tuned model weights
- `tokenizer_config.json`: Tokenizer settings
- `vocab.txt`: Vocabulary file
- `tokenizer.json`: Fast tokenizer configuration
- `training_info.json`: Training metrics and configuration for reference

---

## Usage

### Starting the Application

```bash
streamlit run app.py
```

The application opens in your default browser at `http://localhost:8501`.

### Single Review Analysis

1. Select "Single Review" from the navigation
2. Enter a product review in the text area, or click a sample review button
3. Click "Analyze Review"
4. View the results:
   - Product identification (auto-detected category and brand)
   - Sentiment analysis from three methods (VADER, fine-tuned model, combined)
   - Key insights (pros and cons)
   - Aspect-based analysis with confidence scores
   - Raw JSON data (expandable section)

### Batch Analysis

1. Select "Batch Analysis" from the navigation
2. Choose an input method:
   - **Paste Reviews**: Enter reviews one per line
   - **Upload CSV**: Upload a CSV file with a "Review Text" column
   - **Samples**: Load pre-built sample reviews for testing
3. Optionally enter a product name (the system will auto-detect if left empty)
4. Click "Analyze All Reviews"
5. View the results:
   - Product identification card
   - Verdict (Strongly Recommended / Recommended with Reservations / Mixed Reviews / Not Recommended / Avoid)
   - Overview statistics (total reviews, positive percentage, neutral percentage, negative percentage)
   - Sentiment distribution from VADER and fine-tuned model
   - Key insights (aggregated pros and cons)
   - Warnings (aspects with high negative sentiment)
   - Aspect breakdown (mention count, positive/neutral/negative percentages per aspect)
   - Raw JSON data (expandable section)

### Verdict Logic

| Condition | Verdict |
|-----------|---------|
| Positive >= 70% | Strongly Recommended |
| Positive >= 50% | Recommended with Reservations |
| Negative >= 70% | Not Recommended |
| Negative >= 50% | Avoid |
| Otherwise | Mixed Reviews |

### Warning Logic

An aspect triggers a warning when:
- Negative sentiment percentage is >= 60%
- The aspect is mentioned in >= 10% of total reviews

---

## Technical Details

### Models Used

| Model | Purpose | Source | Size |
|-------|---------|--------|------|
| DistilBERT-base-uncased | Base for fine-tuning | Hugging Face | ~260 MB |
| Fine-tuned DistilBERT | Sentiment classification | Trained on your data | ~260 MB |
| BART-large-mnli | Zero-shot aspect extraction | Hugging Face | ~1.6 GB |
| spaCy en_core_web_sm | Sentence splitting, NER | spaCy | ~13 MB |
| VADER | Rule-based sentiment | vaderSentiment | Built-in |

### Aspect Keyword Mapping

The system uses 23 predefined product aspects, each with multiple keyword variations:

| Aspect | Example Keywords |
|--------|-----------------|
| Battery Life | battery, charge, charging, hours, battery drain |
| Sound Quality | sound, audio, bass, treble, clarity |
| Build Quality | build, construction, sturdy, flimsy, plastic, crack |
| Comfort | comfortable, ear cushion, clamping, fit |
| Connectivity | bluetooth, connection, disconnect, pairing |
| Microphone Quality | microphone, mic, call quality, voice |
| Noise Cancellation | noise cancellation, anc, blocking noise |
| Software | app, software, firmware, settings |
| Value for Money | value, worth, price, expensive, cheap |
| Design | design, look, aesthetics, appearance |
| Ease of Use | easy to use, controls, touch controls |
| Performance | performance, fast, slow, lag, responsive |
| Display | display, screen, brightness, resolution |
| Camera | camera, photos, video quality |
| Customer Service | customer service, support, warranty |
| Durability | durable, broke, broken, damage |
| Packaging | packaging, box, unboxing |
| Delivery | delivery, shipping, arrived |
| Size | size, compact, bulky, small, large |
| Weight | weight, heavy, light, lightweight |
| Material Quality | material, leather, metal, fabric |
| Reliability | reliable, issues, problems |
| User Interface | ui, interface, menu, navigation |

### Aspect Extraction Method

The system uses a two-stage approach:

**Stage 1: Keyword Matching**
- Scans review text for all keywords associated with each aspect
- Counts occurrences and calculates a confidence score based on frequency
- More keyword matches = higher confidence

**Stage 2: Zero-shot Classification**
- Sends the first 500 characters of the review to BART-large-mnli
- Classifies against all 23 aspect labels
- Filters results with confidence > 0.3

**Merging**
- If both methods detect the same aspect, confidence is averaged
- If only one method detects it, that confidence is used
- Results are sorted by confidence in descending order

### Aspect Sentiment Method

1. Split the review into sentences using spaCy
2. For each detected aspect, find sentences containing any of its keywords
3. Run VADER sentiment analysis on each relevant sentence
4. Classify each sentence as positive (compound >= 0.05), negative (compound <= -0.05), or neutral
5. Determine overall aspect sentiment by majority vote
6. Calculate sentiment confidence as the ratio of the majority class to total relevant sentences

### Sentiment Confidence Calculation

For the combined sentiment score:
```
Combined Confidence = (VADER Confidence + Custom Model Confidence) / 2
```

Where:
- VADER Confidence = absolute value of compound score (0 to 1)
- Custom Model Confidence = softmax probability from DistilBERT (0 to 1)

---

## Evaluation

### Metrics Calculated

The training script computes the following metrics on the test set:

| Metric | Definition |
|--------|------------|
| Accuracy | Fraction of correctly classified reviews |
| Precision | Weighted average precision across all classes |
| Recall | Weighted average recall across all classes |
| F1 Score | Weighted average F1 across all classes |
| Confusion Matrix | Per-class breakdown of predictions vs actual labels |

### Interpreting Results

- **Accuracy > 0.85**: Good model performance
- **Accuracy > 0.90**: Strong model performance
- **F1 > 0.85**: Good balance between precision and recall
- **Class Imbalance**: If your dataset has skewed distribution (e.g., 68% negative), the model may be biased toward the majority class. The weighted F1 score accounts for this.

### Class Distribution Considerations

The current Amazon dataset has an imbalanced distribution:
- Negative: 68.2%
- Neutral: 4.2%
- Positive: 27.6%

This means:
- The model may over-predict negative sentiment
- Neutral class has few examples, so performance may be lower
- Consider collecting more balanced data or using class weights for improvement

---

## Dataset

### Source

The dataset used is an Amazon product reviews CSV with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| Reviewer Name | String | Name of the reviewer |
| Profile Link | String | URL to reviewer profile |
| Country | String | Reviewer's country |
| Review Count | String | Number of reviews by this reviewer |
| Review Date | String | Date of the review (ISO format) |
| Rating | String | Rating in format "Rated X out of 5 stars" |
| Review Title | String | Title of the review |
| Review Text | String | Full review body |
| Date of Experience | String | Date of the purchase experience |

### Data Cleaning Applied

The training script performs the following cleaning:
- Removes rows with missing Review Text or Rating
- Extracts numeric rating from text format (e.g., "Rated 1 out of 5 stars" becomes 1)
- Filters to valid ratings (1-5 only)
- Combines Review Title and Review Text for richer input
- Maps ratings to three sentiment classes
- Removes reviews shorter than 10 characters
- Handles CSV encoding issues with multiple fallback encodings

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total rows | 21,214 |
| After cleaning | 21,055 |
| Training set | 16,844 (80%) |
| Test set | 4,211 (20%) |
| Negative samples | 14,350 (68.2%) |
| Neutral samples | 885 (4.2%) |
| Positive samples | 5,820 (27.6%) |

---

## Dependencies

### Required Packages

| Package | Purpose |
|---------|---------|
| streamlit | Web application framework |
| transformers | Pre-trained models and training utilities |
| torch | Deep learning backend |
| spacy | NLP processing (sentence splitting, NER) |
| vaderSentiment | Rule-based sentiment analysis |
| pandas | Data manipulation |
| scikit-learn | Metrics and data splitting |
| datasets | Hugging Face dataset utilities |
| accelerate | Training optimization |

### Model Downloads

On first run, the following models are downloaded and cached:

| Model | Size | Downloaded By |
|-------|------|---------------|
| distilbert-base-uncased | ~260 MB | train_model.py |
| bart-large-mnli | ~1.6 GB | nlp_engine.py / batch_engine.py |
| en_core_web_sm | ~13 MB | spaCy |

Total first-time download: approximately 2 GB. Subsequent runs use cached models.

### Python Version

Tested on Python 3.9+. Compatible with Python 3.14 (with some transformer warnings that do not affect functionality).

---

## Future Work

### Planned Features

1. **Fake Review Detection**: Identify potentially fake or paid reviews using linguistic pattern analysis
2. **Temporal Sentiment Analysis**: Track how sentiment changes over time using review dates
3. **Personalized Recommendations**: Weight aspects based on user priorities to generate personalized verdicts
4. **Reviewer Credibility Scoring**: Score reviewer trustworthiness based on review patterns
5. **Emotion Detection**: Classify emotions (anger, joy, disappointment) beyond positive/negative
6. **Comparative Analysis**: Compare multiple products side by side
7. **Review Summarization**: Generate natural language summaries using LLMs
8. **Export Functionality**: Download results as PDF or image

### Model Improvements

1. **Class Weighting**: Address dataset imbalance with weighted loss function
2. **Larger Base Model**: Experiment with BERT or RoBERTa instead of DistilBERT
3. **Aspect Sentiment Model**: Train a dedicated aspect-level sentiment model instead of rule-based approach
4. **Multi-label Classification**: Allow reviews to have multiple sentiment labels simultaneously

---

## References

- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
- Lewis, P., et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation.
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
- Honnibal, M. & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings.

---

## License

This project is developed for educational and research purposes. The dataset used for training consists of publicly available Amazon product reviews. The pre-trained models are used under their respective licenses from Hugging Face.
