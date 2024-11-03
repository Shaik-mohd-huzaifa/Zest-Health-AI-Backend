from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from training_data import train_data
import joblib
import os
import pandas as pd


def preprocess_text(text):
    return text.lower()


def train_initial_model():
    df = pd.DataFrame(train_data, columns=["text", "label"])
    df["text"] = df["text"].apply(preprocess_text)

    train_x = df["text"]
    train_y = df["label"]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_df=0.85, min_df=2, max_features=5000
    )
    X_vectorized = vectorizer.fit_transform(train_x)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, train_y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    # Use StratifiedKFold with only 3 splits
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=stratified_kfold,
        scoring="accuracy",
        verbose=2,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(best_model, "./ZestHealth/trained_nlp_model/model.pkl")
    joblib.dump(vectorizer, "./ZestHealth/trained_nlp_model/vectorizer.pkl")


train_initial_model()


def update_model(new_query, new_label):
    model_path = "./ZestHealth/trained_nlp_model/model.pkl"
    vectorizer_path = "./ZestHealth/trained_nlp_model/vectorizer.pkl"

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    else:
        print("Model or vectorizer not found. Please train the initial model first.")
        return

    new_query_vectorized = vectorizer.transform([new_query])

    train_x = [text for text, label in train_data]
    train_y = [label for text, label in train_data]

    train_x.append(new_query)
    train_y.append(new_label)

    X_vectorized = vectorizer.fit_transform(train_x)
    model.fit(X_vectorized, train_y)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("New Query Updated")
