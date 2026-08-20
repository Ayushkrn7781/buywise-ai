"""
Train a sentiment classification model on your Amazon Reviews dataset.

Usage:
    python train_model.py --data Amazon_Reviews.csv --epochs 3
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import argparse
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================

LABEL_MAP = {
    1: 0,  # 1 star -> Negative
    2: 0,  # 2 star -> Negative
    3: 1,  # 3 star -> Neutral
    4: 2,  # 4 star -> Positive
    5: 2   # 5 star -> Positive
}

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
SAVED_MODEL_PATH = "./trained_sentiment_model"


# ==================== DATA LOADING ====================

def load_data(csv_path):
    """Load and preprocess the Amazon reviews CSV"""
    print(f"Loading data from {csv_path}...")
    
    # Use Python parser (slower but handles messy CSVs)
    df = pd.read_csv(
        csv_path,
        encoding='utf-8',
        engine='python',           # This fixes the buffer overflow
        on_bad_lines='skip',
        quoting=1                  # QUOTE_ALL for safety
    )
    
    print(f"Total rows loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check required columns
    required = ['Review Text', 'Rating']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
    
    # Remove rows with missing Review Text or Rating
    df = df.dropna(subset=['Review Text', 'Rating'])
    
    # Convert Review Text to string
    df['Review Text'] = df['Review Text'].astype(str)
    
    # Fix Rating: "Rated 1 out of 5 stars" -> 1
    df['Rating'] = df['Rating'].astype(str)
    df['Rating'] = df['Rating'].str.extract(r'(\d+)')  # Extract first number
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df.dropna(subset=['Rating'])
    df['Rating'] = df['Rating'].astype(int)
    
    # Filter valid ratings (1-5)
    df = df[df['Rating'].between(1, 5)]
    
    # Combine Review Title + Review Text
    if 'Review Title' in df.columns:
        df['Review Title'] = df['Review Title'].fillna('').astype(str)
        df['full_text'] = df['Review Title'] + '. ' + df['Review Text']
    else:
        df['full_text'] = df['Review Text']
    
    # Map rating to sentiment label
    df['label'] = df['Rating'].map(LABEL_MAP)
    
    # Remove invalid labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    # Basic cleaning
    df['full_text'] = df['full_text'].str.strip()
    df = df[df['full_text'].str.len() > 10]
    
    print(f"Clean dataset size: {len(df)}")
    print(f"\nLabel distribution:")
    for label_id, label_name in ID2LABEL.items():
        count = (df['label'] == label_id).sum()
        pct = count / len(df) * 100
        print(f"  {label_name}: {count} ({pct:.1f}%)")
    
    return df[['full_text', 'label', 'Rating']]

# ==================== DATASET CLASS ====================

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        encoding['labels'] = label
        return encoding


# ==================== METRICS ====================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ==================== TRAINING ====================

def train(csv_path, epochs=3, batch_size=16, test_size=0.2):
    """Main training function"""
    
    # Load data
    df = load_data(csv_path)
    
    # Split data
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['label'],
        random_state=42
    )
    
    print(f"\nTrain size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Load tokenizer and model
    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    # Create datasets
    train_dataset = ReviewDataset(
        train_df['full_text'].values,
        train_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    test_dataset = ReviewDataset(
        test_df['full_text'].values,
        test_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
      # Calculate warmup steps
    total_steps = (len(train_dataset) // batch_size) * epochs
    warmup_steps = int(total_steps * 0.1)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    trainer.train()
    
    # Evaluate
    print("\n" + "="*50)
    print("EVALUATION ON TEST SET")
    print("="*50)
    
    eval_results = trainer.evaluate()
    print(f"\nAccuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"Precision: {eval_results['eval_precision']:.4f}")
    print(f"Recall:    {eval_results['eval_recall']:.4f}")
    print(f"F1 Score:  {eval_results['eval_f1']:.4f}")
    
    # Confusion matrix
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    cm = confusion_matrix(test_df['label'].values, preds)
    
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("           Neg   Neu   Pos")
    for i, row in enumerate(cm):
        label = ID2LABEL[i][:3].upper()
        print(f"Actual {label}  {row[0]:5d} {row[1]:5d} {row[2]:5d}")
    
    # Save model
    print(f"\nSaving model to {SAVED_MODEL_PATH}...")
    trainer.save_model(SAVED_MODEL_PATH)
    tokenizer.save_pretrained(SAVED_MODEL_PATH)
    
    # Save training info
    info = {
        "model_name": MODEL_NAME,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": MAX_LENGTH,
        "accuracy": eval_results['eval_accuracy'],
        "precision": eval_results['eval_precision'],
        "recall": eval_results['eval_recall'],
        "f1": eval_results['eval_f1']
    }
    
    with open(os.path.join(SAVED_MODEL_PATH, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Model saved to: {SAVED_MODEL_PATH}")
    
    return eval_results


# ==================== MAIN ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment model on Amazon reviews")
    parser.add_argument("--data", type=str, default="Amazon_Reviews.csv", 
                        help="Path to Amazon reviews CSV")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="Test set fraction")
    
    args = parser.parse_args()
    
    train(
        csv_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size
    )