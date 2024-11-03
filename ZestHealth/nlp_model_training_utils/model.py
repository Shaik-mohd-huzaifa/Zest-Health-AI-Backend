# your_app/ml_models/model.py

import joblib


def load_model(model_path):
    return joblib.load(model_path)


def load_vectorizer(vectorizer_path):
    return joblib.load(vectorizer_path)
