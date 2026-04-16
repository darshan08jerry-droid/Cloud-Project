import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.preprocess import preprocess_text

# Dataset URL
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
DATA_ZIP = 'data/smsspamcollection.zip'
DATA_FILE = 'data/SMSSpamCollection'

def download_and_extract_data():
    if not os.path.exists(DATA_FILE):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_ZIP)
        print("Extracting dataset...")
        with zipfile.ZipFile(DATA_ZIP, 'r') as zip_ref:
            zip_ref.extractall('data/')
        os.remove(DATA_ZIP)
    else:
        print("Dataset already exists.")

def load_data():
    print("Loading data...")
    # The file is tab-separated with columns: label, message
    df = pd.read_csv(DATA_FILE, sep='\t', header=None, names=['label', 'message'])
    
    # Map labels to 'Spam' and 'Not Spam'
    df['label'] = df['label'].map({'ham': 'Not Spam', 'spam': 'Spam'})
    return df

def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    
    # Needs pos_label because strings are used instead of 0/1
    prec = precision_score(y_true, y_pred, pos_label='Spam')
    rec = recall_score(y_true, y_pred, pos_label='Spam')
    f1 = f1_score(y_true, y_pred, pos_label='Spam')
    
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def main():
    # 1. Download and Load Data
    download_and_extract_data()
    df = load_data()
    
    # 2. Preprocess Text
    print("Preprocessing text (this may take a minute)...")
    # Apply preprocessing from our module
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    # Drop rows that somehow became empty after preprocessing
    df = df[df['processed_message'] != ""]
    
    # Feature & Label extraction
    X = df['processed_message']
    y = df['label']
    
    # Train-test split
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. TF-IDF Vectorization
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 4. Model Training
    print("Training models...")
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_f1 = 0
    best_model_name = ""
    best_model = None
    
    # 5. Model Evaluation
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        metrics = evaluate_model(y_test, y_pred, name)
        results[name] = metrics
        
        # Select best model based on F1 Score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_name = name
            best_model = model

    print(f"=====================================")
    print(f"Best Model: {best_model_name} (F1: {best_f1:.4f})")
    print(f"=====================================")

    # 6. Model Saving
    print("Saving the best model and vectorizer...")
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Model and vectorizer saved successfully to 'model/' directory.")

if __name__ == "__main__":
    main()
