import pickle
import os
from src.preprocess import preprocess_text

def load_artifacts():
    '''
    Loads the trained model and TF-IDF vectorizer from disk.
    '''
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
    vec_path = os.path.join(os.path.dirname(__file__), 'model', 'vectorizer.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

def predict_spam(text):
    '''
    Takes raw email text, applies preprocessing, vectorization,
    and returns the prediction alongside its confidence probability.
    '''
    if not text.strip():
        return "Not Spam", 0.0

    try:
        model, vectorizer = load_artifacts()
    except FileNotFoundError:
        return "Error: Model artifacts not found. Please train the model first.", 0.0

    # 1. Preprocess text
    processed_text = preprocess_text(text)
    
    # Check if empty after preprocessing
    if not processed_text:
         return "Not Spam", 0.0

    # 2. Vectorize
    vectorized_text = vectorizer.transform([processed_text])
    
    # 3. Predict
    prediction = model.predict(vectorized_text)[0]
    
    # Get probability
    # model.classes_ will output ['Not Spam', 'Spam'] typically based on alphabetical order, 
    # but let's check correctly by probability index
    probs = model.predict_proba(vectorized_text)[0]
    
    # Get confidence of the chosen prediction
    pred_index = list(model.classes_).index(prediction)
    confidence = probs[pred_index]
    
    return prediction, confidence
