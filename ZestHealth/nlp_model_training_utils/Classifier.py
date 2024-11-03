# your_app/ml_models/classifier.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from .preprocessing import preprocess_input

def classify_input(query, vectorizer, model):
    query = preprocess_input(query)
    query_vector = vectorizer.transform([query])
    prediction = model.predict(query_vector)
    print(f"Prediction: {prediction}")
    return prediction[0]


# def classify_input(input_text, vectorizer, model):
#     preprocessed_text = preprocess_input(input_text)
#     print(f"Preprocessed text: {preprocessed_text}")

#     transformed_text = vectorizer.transform([preprocessed_text])
#     print(f"Transformed input: {transformed_text}")

#     if transformed_text.nnz == 0:  # Check if no elements are stored
#         print("Input does not match the vectorizer vocabulary.")

#     predicted_intent = model.predict(transformed_text)[0]
#     print(f"Predicted intent: {predicted_intent}")

#     return predicted_intent




# Note: Ensure to handle the model loading appropriately.
