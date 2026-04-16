import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()

def preprocess_text(text):
    '''
    Preprocesses the input text:
    - Lowercases text
    - Tokenizes
    - Removes punctuation and special characters
    - Removes stopwords
    - Applies stemming
    '''
    if not isinstance(text, str):
        return ""

    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Tokenization
    try:
        tokens = word_tokenize(text)
    except Exception:
        # Fallback if punkt fails for some reason
        tokens = text.split()
    
    # 3. Remove punctuation & special characters
    # Only keep alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum()]
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Apply stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)
