import re
import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    return text.lower()

def preprocess_data(df, text_column="text", fit_vectorizer=True, vectorizer=None):
    df["clean_text"] = df[text_column].apply(clean_text)
    if fit_vectorizer:
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df["clean_text"]).toarray()
        return X, vectorizer
    else:
        X = vectorizer.transform(df["clean_text"]).toarray()
        return X, vectorizer

def train_model():
    if not os.path.exists("data/fake_news.csv"):
        logging.error("Dataset not found at data/fake_news.csv")
        sys.exit(1)
    df = pd.read_csv("data/fake_news.csv")
    X, vectorizer = preprocess_data(df, "text", fit_vectorizer=True)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logging.info("Accuracy: %s", accuracy_score(y_test, y_pred))
    logging.info("Precision: %s", precision_score(y_test, y_pred))
    logging.info("Recall: %s", recall_score(y_test, y_pred))
    logging.info("F1: %s", f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    joblib.dump(model, "fake_news_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    logging.info("Model and vectorizer saved.")

app = Flask(__name__)

try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except:
    model, vectorizer = None, None

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            df = pd.read_csv(file)
            X, _ = preprocess_data(df, "text", fit_vectorizer=False, vectorizer=vectorizer)
            predictions = model.predict(X)
            df["prediction"] = predictions
            return df.to_html()
    return '''
    <h1>Fake News Detector</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload & Predict">
    </form>
    '''

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    texts = data.get("texts", [])
    df = pd.DataFrame({"text": texts})
    X, _ = preprocess_data(df, "text", fit_vectorizer=False, vectorizer=vectorizer)
    predictions = model.predict(X)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model()
    else:
        app.run(debug=True)
