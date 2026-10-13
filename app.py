import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="GRU Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_gru_model():

    model = tf.keras.models.load_model(
        "best_gru_model.keras"
    )

    return model


# ==========================================
# LOAD TOKENIZER
# ==========================================

@st.cache_resource
def load_tokenizer():

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    return tokenizer


# ==========================================
# LOAD MAX LENGTH
# ==========================================

@st.cache_resource
def load_max_length():

    with open("max_len.pkl", "rb") as file:
        max_len = pickle.load(file)

    return max_len


model = load_gru_model()
tokenizer = load_tokenizer()
max_sequence_length = load_max_length()


# ==========================================
# CLEAN TEXT
# ==========================================

def clean_text(text):

    text = str(text)
    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# INDEX → WORD
# ==========================================

index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}


# ==========================================
# SAMPLE WORD
# ==========================================

def sample_word(
    predictions,
    temperature=0.7,
    top_k=10
):

    predictions = np.asarray(
        predictions
    ).astype("float64")

    predictions = np.log(
        predictions + 1e-10
    )

    predictions = predictions / temperature

    predictions = np.exp(
        predictions
    )

    predictions = predictions / np.sum(
        predictions
    )

    top_indices = np.argsort(
        predictions
    )[-top_k:]

    top_probs = predictions[
        top_indices
    ]

    top_probs = top_probs / np.sum(
        top_probs
    )

    predicted_id = np.random.choice(
        top_indices,
        p=top_probs
    )

    return predicted_id


# ==========================================
# GENERATE TEXT
# ==========================================

def generate_text(
    seed_text,
    next_words,
    temperature,
    top_k
):

    generated_text = clean_text(
        seed_text
    )

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences(
            [generated_text]
        )[0]

        if len(token_list) == 0:
            break

        token_list = token_list[
            -(max_sequence_length - 1):
        ]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_length - 1,
            padding="pre"
        )

        predictions = model.predict(
            token_list,
            verbose=0
        )[0]

        predicted_id = sample_word(
            predictions,
            temperature=temperature,
            top_k=top_k
        )

        predicted_word = index_to_word.get(
            predicted_id
        )

        if predicted_word is None:
            break

        generated_text += " " + predicted_word

    return generated_text


# ==========================================
# STREAMLIT UI
# ==========================================

st.title("🧠 GRU Next Word Predictor")

st.write(
    "Enter a sentence and let the GRU model "
    "generate the next words."
)


# ==========================================
# USER INPUT
# ==========================================

text_input = st.text_area(
    "Enter your text:",
    placeholder="Example: The purpose of",
    height=100
)


# ==========================================
# SETTINGS
# ==========================================

col1, col2 = st.columns(2)

with col1:

    temperature = st.slider(
        "Temperature",
        min_value=0.2,
        max_value=1.2,
        value=0.7,
        step=0.1
    )

with col2:

    next_words = st.slider(
        "Words to generate",
        min_value=1,
        max_value=20,
        value=8
    )


top_k = st.slider(
    "Top-K",
    min_value=3,
    max_value=30,
    value=10
)


# ==========================================
# GENERATE BUTTON
# ==========================================

if st.button(
    "🚀 Generate",
    use_container_width=True
):

    if text_input.strip() == "":

        st.warning(
            "Please enter some text first."
        )

    else:

        result = generate_text(
            text_input,
            next_words,
            temperature,
            top_k
        )

        st.subheader("Generated Text")

        st.success(result)