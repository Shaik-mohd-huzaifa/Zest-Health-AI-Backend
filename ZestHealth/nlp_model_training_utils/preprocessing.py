# your_app/ml_models/preprocessing.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("punkt")
nltk.download("punkt_tab")


# def preprocess_input(text):
#     tokens = word_tokenize(text)  # Tokenize the input text
#     stemmer = PorterStemmer()  # Initialize the PorterStemmer
#     stems = [stemmer.stem(token) for token in tokens]  # Stem each token
#     return " ".join(stems)  # Join the stemmed tokens into a single string


lemmatizer = WordNetLemmatizer()


def preprocess_input(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return " ".join(tokens)
