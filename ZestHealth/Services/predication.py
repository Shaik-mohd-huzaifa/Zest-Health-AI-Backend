import joblib
from ZestHealth.nlp_model_training_utils.Classifier import classify_input

model = joblib.load("./trained_nlp_model/model.pkl")
vectorizer = joblib.load("./trained_nlp_model/vectorizer.pkl")


def prediction(query):
    intent = classify_input(query, model=model, vectorizer=vectorizer)
    return intent
