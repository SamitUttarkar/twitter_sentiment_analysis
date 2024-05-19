from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import spacy
import pandas as pd
import joblib  # Use joblib instead of pickle
import base64
import matplotlib.pyplot as plt

# Load SpaCy model and stop words
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

app = Flask(__name__)

# Define the cleaning and lemmatization function
def clean_and_lemmatize(content):
    doc = nlp(content)
    lemmatized_content = [token.lemma_ for token in doc if token.text not in spacy_stopwords and token.is_alpha]
    return ' '.join(lemmatized_content)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load model and vectorizer using joblib
        predictor = joblib.load(r"Models/dt_classifier.pkl")
        tfidf_vectorizer = joblib.load(r"Models/tfidfvectorizer.pkl")

        if "file" in request.files:
            # Bulk prediction
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions, graph = bulk_prediction(predictor, tfidf_vectorizer, data)
            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )
            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("ascii")
            return response

        elif "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, tfidf_vectorizer, text_input)
            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})

def single_prediction(predictor, tfidf_vectorizer, text_input):
    cleaned_text = clean_and_lemmatize(text_input)
    X_prediction = tfidf_vectorizer.transform([cleaned_text]).toarray()
    y_predictions = predictor.predict_proba(X_prediction)
    y_predictions = y_predictions.argmax(axis=1)[0]
    return "Positive" if y_predictions == 1 else "Negative"

def bulk_prediction(predictor, tfidf_vectorizer, data):
    data["cleaned_text"] = data["text"].apply(clean_and_lemmatize)
    X_prediction = tfidf_vectorizer.transform(data["cleaned_text"]).toarray()
    y_predictions = predictor.predict_proba(X_prediction)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))
    data["Predicted sentiment"] = y_predictions

    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)
    graph = get_distribution_graph(data)
    return predictions_csv, graph

def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)
    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )
    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()
    return graph

def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug=True)

