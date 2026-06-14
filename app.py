from flask import Flask, render_template, request
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download('punkt_tab')

app = Flask(__name__)

# -------------------------
# TRAIN MODEL
# -------------------------
def train_model():

    dataset = load_dataset(
    "dair-ai/emotion",
    split="train[:5000]"
    )

    df = pd.DataFrame(dataset)

    sentiment_map = {
        0: "Negative",
        1: "Positive",
        2: "Positive",
        3: "Negative",
        4: "Negative",
        5: "Neutral"
    }

    df["sentiment"] = df["label"].map(sentiment_map)

    X = df["text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    return model, accuracy

model, accuracy = train_model()

# -------------------------
# HOME PAGE
# -------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    tokens = None

    if request.method == "POST":

        text = request.form["review"]

        if text.strip():

            tokens = word_tokenize(text)

            prediction = model.predict([text])[0]

            confidence = max(
                model.predict_proba([text])[0]
            )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        tokens=tokens,
        accuracy=accuracy
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )