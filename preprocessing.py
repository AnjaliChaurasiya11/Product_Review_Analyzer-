import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# We need to download nltk data if not already present
# We will handle this gracefully in the main setup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    # punkt_tab is required for newer versions of nltk
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def clean_text(text):
    '''
    Preprocess the text:
    - Lowercase
    - Remove punctuation and special characters
    - Remove stopwords
    '''
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation & special characters using regex
    # We keep words and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    # Optional: Keep words that might be important for sentiment like 'not', 'no'
    # For a beginner project, standard stopwords removal is fine
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(filtered_tokens)

