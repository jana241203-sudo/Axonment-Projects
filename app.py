from __future__ import annotations

import io
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


# Keep the page configuration at the top of the Streamlit script.
st.set_page_config(page_title="Amazon Review Sentiment", page_icon="🛍️", layout="wide")

SENTIMENT_NAMES = ["Negative", "Neutral", "Positive"]
CLASS_COLORS = {"Negative": "#E76F51", "Neutral": "#E9C46A", "Positive": "#2A9D8F"}
VOCABULARY_SIZE = 20_000
MAX_SEQUENCE_LENGTH = 180


def clean_review(review: str) -> str:
    """Normalize review text while retaining letters and useful word order."""
    review = str(review).lower()
    review = re.sub(r"<br\s*/?>", " ", review)       # Remove HTML line breaks.
    review = re.sub(r"<[^>]+>", " ", review)         # Remove any remaining HTML tags.
    review = re.sub(r"[^a-z0-9\s']", " ", review)   # Remove punctuation and symbols.
    return re.sub(r"\s+", " ", review).strip()      # Collapse repeated whitespace.


def score_to_sentiment(score: int) -> str:
    """Convert Amazon's 1-5 star rating into the project's three labels."""
    if score <= 2:
        return "Negative"
    if score == 3:
        return "Neutral"
    return "Positive"


@st.cache_data(show_spinner=False)
def load_balanced_reviews(file_bytes: bytes, reviews_per_class: int) -> pd.DataFrame:
    """Read a CSV in chunks and create a balanced sample without loading it all at once."""
    sampled_chunks = []
    csv_stream = io.BytesIO(file_bytes)

    # Kaggle's Amazon Fine Food Reviews file uses the columns Score and Text.
    for chunk in pd.read_csv(csv_stream, usecols=["Score", "Text"], chunksize=25_000):
        chunk = chunk.dropna(subset=["Score", "Text"]).copy()
        chunk["Score"] = pd.to_numeric(chunk["Score"], errors="coerce")
        chunk = chunk.dropna(subset=["Score"])
        chunk = chunk[chunk["Score"].between(1, 5)]
        chunk["Sentiment"] = chunk["Score"].astype(int).map(score_to_sentiment)

        # Keep a bounded sample from each chunk while supporting smaller prepared CSVs.
        chunk_sample_limit = max(700, min(reviews_per_class, 1_500))
        for sentiment in SENTIMENT_NAMES:
            class_rows = chunk[chunk["Sentiment"] == sentiment]
            if not class_rows.empty:
                sampled_chunks.append(
                    class_rows.sample(n=min(len(class_rows), chunk_sample_limit), random_state=42)
                )

    if not sampled_chunks:
        raise ValueError("No usable reviews found. Check that the CSV has Score and Text columns.")

    candidates = pd.concat(sampled_chunks, ignore_index=True)
    balanced_parts = []

    # Downsample the candidate pool so every sentiment has equal representation.
    for sentiment in SENTIMENT_NAMES:
        class_rows = candidates[candidates["Sentiment"] == sentiment]
        take = min(len(class_rows), reviews_per_class)
        if take == 0:
            raise ValueError(f"The uploaded data has no usable {sentiment.lower()} reviews.")
        balanced_parts.append(class_rows.sample(n=take, random_state=42))

    result = pd.concat(balanced_parts, ignore_index=True)
    result["CleanText"] = result["Text"].map(clean_review)
    return result[result["CleanText"].str.len() > 0].reset_index(drop=True)


def prepare_lstm_text(train_text, test_text):
    """Learn a word vocabulary from training text, then vectorize both splits."""
    text_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCABULARY_SIZE,
        output_mode="int",
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        standardize=None,  # Text is already cleaned above.
    )
    text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_text).batch(256))
    x_train = text_vectorizer(tf.constant(train_text)).numpy()
    x_test = text_vectorizer(tf.constant(test_text)).numpy()
    return text_vectorizer, x_train, x_test


@st.cache_resource(show_spinner=False)
def train_models(file_bytes: bytes, reviews_per_class: int, lstm_epochs: int):
    """Train TF-IDF baselines and an LSTM; return models and held-out metrics."""
    reviews = load_balanced_reviews(file_bytes, reviews_per_class)
    encoder = LabelEncoder()
    encoder.fit(SENTIMENT_NAMES)

    text = reviews["CleanText"].to_numpy()
    labels = encoder.transform(reviews["Sentiment"])
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        text,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    # Fit TF-IDF only on training reviews so test information cannot leak into training.
    tfidf = TfidfVectorizer(max_features=15_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train_tfidf = tfidf.fit_transform(x_train_text)
    x_test_tfidf = tfidf.transform(x_test_text)

    naive_bayes = MultinomialNB(alpha=0.5).fit(x_train_tfidf, y_train)
    logistic = LogisticRegression(max_iter=1_000, class_weight="balanced").fit(
        x_train_tfidf, y_train
    )

    # Prepare token sequences and train a small recurrent neural network.
    text_vectorizer, x_train_lstm, x_test_lstm = prepare_lstm_text(x_train_text, x_test_text)
    tf.keras.utils.set_random_seed(42)
    lstm = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,)),
        tf.keras.layers.Embedding(VOCABULARY_SIZE, 64, mask_zero=True),
        tf.keras.layers.LSTM(48, dropout=0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    lstm.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    history = lstm.fit(
        x_train_lstm,
        y_train,
        validation_split=0.2,
        epochs=lstm_epochs,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0,
    )

    # Score all models on the same held-out test split.
    predictions = {
        "Naïve Bayes": naive_bayes.predict(x_test_tfidf),
        "Logistic Regression": logistic.predict(x_test_tfidf),
        "LSTM": np.argmax(lstm.predict(x_test_lstm, verbose=0), axis=1),
    }
    metrics = {
        name: {
            "accuracy": accuracy_score(y_test, model_predictions),
            "macro_f1": f1_score(y_test, model_predictions, average="macro"),
            "confusion_matrix": confusion_matrix(y_test, model_predictions, labels=[0, 1, 2]),
            "prediction_share": np.bincount(model_predictions, minlength=3) / len(model_predictions),
        }
        for name, model_predictions in predictions.items()
    }

    return {
        "reviews": reviews,
        "tfidf": tfidf,
        "naive_bayes": naive_bayes,
        "logistic": logistic,
        "text_vectorizer": text_vectorizer,
        "lstm": lstm,
        "encoder": encoder,
        "metrics": metrics,
        "history": history.history,
    }


def predict_custom_review(review: str, trained):
    """Run all trained models on one new review and return readable predictions."""
    cleaned = clean_review(review)
    tfidf_features = trained["tfidf"].transform([cleaned])
    sequence = trained["text_vectorizer"](tf.constant([cleaned])).numpy()

    class_ids = {
        "Naïve Bayes": int(trained["naive_bayes"].predict(tfidf_features)[0]),
        "Logistic Regression": int(trained["logistic"].predict(tfidf_features)[0]),
        "LSTM": int(np.argmax(trained["lstm"].predict(sequence, verbose=0)[0])),
    }
    return {name: trained["encoder"].inverse_transform([class_id])[0] for name, class_id in class_ids.items()}


#Streamlit interface 
st.title("🛍️ Amazon Product Review Sentiment")
st.write(
    "Explore customer feedback as **negative**, **neutral**, or **positive**. "
    "Compare two TF-IDF models with an LSTM and try your own review."
)

with st.sidebar:
    st.header("1. Add the dataset")
    uploaded_csv = st.file_uploader("Upload Kaggle Reviews.csv", type=["csv"])
    reviews_per_class = st.select_slider(
        "Reviews to train on per sentiment",
        options=[500, 1_000, 2_000, 3_000, 4_000],
        value=2_000,
    )
    lstm_epochs = st.slider("Maximum LSTM epochs", min_value=2, max_value=8, value=4)
    train_button = st.button("Train / load models", type="primary", use_container_width=True)
    st.caption("Ratings map to labels as follows: 1-2 negative, 3 neutral, 4-5 positive.")

if uploaded_csv is None:
    st.info("Upload the Kaggle Reviews.csv file in the sidebar to begin.")
    st.markdown(
        "Expected columns: **Score** (1-5 stars) and **Text** (review content). "
        "Download: [Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)."
    )
    st.stop()

file_bytes = uploaded_csv.getvalue()
if train_button:
    with st.spinner("Preparing balanced reviews and training the models…"):
        try:
            st.session_state["trained_models"] = train_models(
                file_bytes, reviews_per_class, lstm_epochs
            )
            st.session_state["training_key"] = (uploaded_csv.name, reviews_per_class, lstm_epochs)
        except Exception as error:
            st.error(f"Could not train the models: {error}")
            st.info("Check the CSV columns and install the packages listed in requirements.txt.")

trained = st.session_state.get("trained_models")
active_key = (uploaded_csv.name, reviews_per_class, lstm_epochs)
if trained is None or st.session_state.get("training_key") != active_key:
    st.warning("Click **Train / load models** to train using this file and the selected settings.")
    st.stop()

st.subheader("Model performance on held-out reviews")
metric_columns = st.columns(3)
for column, (model_name, model_metrics) in zip(metric_columns, trained["metrics"].items()):
    column.metric(model_name, f"{model_metrics['accuracy']:.1%} accuracy")
    column.caption(f"Macro F1: {model_metrics['macro_f1']:.3f}")

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(trained["metrics"].keys())
    accuracies = [trained["metrics"][name]["accuracy"] for name in names]
    ax.bar(names, accuracies, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy by model")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    model_for_dashboard = st.selectbox("Sentiment distribution model", list(trained["metrics"].keys()))
    shares = trained["metrics"][model_for_dashboard]["prediction_share"]
    distribution_columns = st.columns(3)
    for column, sentiment, share in zip(distribution_columns, SENTIMENT_NAMES, shares):
        column.metric(f"{sentiment} predictions", f"{share:.1%}")
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.barh(SENTIMENT_NAMES, shares, color=[CLASS_COLORS[name] for name in SENTIMENT_NAMES])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of held-out predictions")
    ax.set_title("Predicted sentiment mix")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Confusion matrix")
matrix_model = st.selectbox("Choose a model to inspect", list(trained["metrics"].keys()))
matrix = trained["metrics"][matrix_model]["confusion_matrix"]
fig, ax = plt.subplots(figsize=(5, 4))
image = ax.imshow(matrix, cmap="Blues")
ax.set_xticks(range(3), labels=SENTIMENT_NAMES, rotation=15)
ax.set_yticks(range(3), labels=SENTIMENT_NAMES)
ax.set_xlabel("Predicted sentiment")
ax.set_ylabel("Actual sentiment")
ax.set_title(f"{matrix_model} confusion matrix")
for row in range(3):
    for col in range(3):
        ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("LSTM training curve")
history = trained["history"]
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(history["accuracy"], label="Training accuracy")
ax.plot(history["val_accuracy"], label="Validation accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Try your own product review")
new_review = st.text_area(
    "Paste a review",
    placeholder="Example: The product arrived quickly and works exactly as expected.",
    height=110,
)
if st.button("Analyze review", type="primary"):
    if not new_review.strip():
        st.warning("Enter a review first.")
    else:
        predictions = predict_custom_review(new_review, trained)
        output_columns = st.columns(3)
        for column, model_name in zip(output_columns, predictions):
            sentiment = predictions[model_name]
            column.metric(model_name, sentiment)

st.caption("Sentiment predictions are estimates. Sarcasm, mixed opinions, and product-specific language can be difficult to classify.")
