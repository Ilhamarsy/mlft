import streamlit as st
import numpy as np
import pickle
import re
import json
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

MAX_LEN = 180

# Define Topic Mapping (0-7) derived from dataset samples
TOPIC_MAPPING = {
    0: "Pengalaman Pengguna & Stabilitas",  # E.g., "mudah dipahami", "tidak error"
    1: "Kepuasan Pengguna",                 # E.g., "puas", "terima kasih"
    2: "Kebermanfaatan Aplikasi",           # E.g., "sangat membantu"
    3: "Edukasi & Investasi",               # E.g., "belajar", "investor", "trader"
    4: "Fitur & Transaksi",                 # E.g., "sell opsional", "triger"
    5: "Pendaftaran & Akun (RDN)",          # E.g., "RDN jadi", "verifikasi"
    6: "Dividen & Keuangan",                # E.g., "pembayaran dividen"
    7: "Kendala Teknis & Error"             # E.g., "gagal", "loading", "bug"
}

st.set_page_config(
    page_title="Aspect-Based Sentiment Analysis",
    layout="centered"
)

st.title("Aspect-Based Sentiment Analysis")
st.write("Analisis sentimen dan aspek pada ulasan aplikasi menggunakan BIGRU + FastText")

@st.cache_resource
def load_models():
    sentiment_model = load_model("sentiment_model.h5")
    aspect_model = load_model("topic_model.h5")
    return sentiment_model, aspect_model

@st.cache_resource
def load_tokenizers():
    with open("tokenizer_sentiment.pkl", "rb") as f:
        tok_sent = pickle.load(f)
    with open("tokenizer_topic.pkl", "rb") as f:
        tok_asp = pickle.load(f)
    return tok_sent, tok_asp

@st.cache_resource
def load_preprocess_resources():
    with open("merged_slang_dict.json", "r", encoding="utf-8") as f:
        slang_dict = json.load(f)

    # Stopwords for Sentiment (Filtered)
    with open("stopwords-id-filtered.txt", "r", encoding="utf-8") as f:
        stopwords_sent = set(f.read().splitlines())
    if "nya" not in stopwords_sent:
        stopwords_sent.add("nya")

    # Stopwords for Topic (Full)
    with open("stopwords-id.txt", "r", encoding="utf-8") as f:
        stopwords_topic = set(f.read().splitlines())
    if "nya" not in stopwords_topic:
        stopwords_topic.add("nya")

    stemmer = StemmerFactory().create_stemmer()

    return slang_dict, stopwords_sent, stopwords_topic, stemmer

sentiment_model, aspect_model = load_models()
tok_sent, tok_asp = load_tokenizers()
slang_dict, stopwords_sent, stopwords_topic, stemmer = load_preprocess_resources()

def cleaning(text: str) -> str:
    text = text.lower()
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[^a-z]", " ", text)
    text = re.sub(r"\t|\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text.strip()


def normalize_slang(text: str) -> str:
    tokens = re.findall(r"\w+|\S", text)
    normalized = [slang_dict.get(tok, tok) for tok in tokens]
    return " ".join(normalized)


def remove_stopwords(text: str, stopwords_set: set) -> str:
    return " ".join([w for w in text.split() if w not in stopwords_set])


def stemming(text: str) -> str:
    return stemmer.stem(text)


def preprocess_sentiment(text: str) -> str:
    """
    Pipeline for Sentiment Model:
    Cleaning -> Slang Normalization -> Stopwords Removal (Filtered) -> (No Stemming)
    """
    text = cleaning(text)
    text = normalize_slang(text)
    text = remove_stopwords(text, stopwords_sent)
    return text


def preprocess_topic(text: str) -> str:
    """
    Pipeline for Topic Model:
    Cleaning -> Slang Normalization -> Stopwords Removal (Full) -> Stemming
    """
    text = cleaning(text)
    text = normalize_slang(text)
    text = remove_stopwords(text, stopwords_topic)
    text = stemming(text)
    return text


def tokenize_and_pad(text: str, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return pad

st.write("---")
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (Upload)"])

with tab1:
    st.subheader("Masukkan Teks Ulasan")
    user_input = st.text_area(
        "Contoh: Tampilan aplikasinya sederhana dan mudah dipahami",
        height=120
    )

    if st.button("Analisis"):
        if user_input.strip() == "":
            st.warning("Teks tidak boleh kosong.")
        else:
            # Preprocessing & Analysis for Sentiment
            text_sent_processed = preprocess_sentiment(user_input)
            X_sent = tokenize_and_pad(text_sent_processed, tok_sent)
            sent_prob = sentiment_model.predict(X_sent, verbose=0)[0][0]
            sent_label = "Positif" if sent_prob >= 0.5 else "Negatif"

            # Preprocessing & Analysis for Topic
            text_topic_processed = preprocess_topic(user_input)
            X_asp = tokenize_and_pad(text_topic_processed, tok_asp)
            asp_probs = aspect_model.predict(X_asp, verbose=0)[0]
            asp_idx = np.argmax(asp_probs)
            asp_conf = asp_probs[asp_idx]
            
            # Get Topic Label from Mapping
            asp_label = TOPIC_MAPPING.get(asp_idx, f"Topic {asp_idx}")

            # Visualization
            st.divider()
            st.caption(f"**Processed Text (Sentiment):** `{text_sent_processed}`")
            st.caption(f"**Processed Text (Topic):** `{text_topic_processed}`")
            
            st.subheader("Hasil Analisis")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sentimen")
                st.info(f"**{sent_label}**")
                st.metric("Confidence", f"{sent_prob:.2%}" if sent_prob >= 0.5 else f"{(1-sent_prob):.2%}")
                
            with col2:
                st.subheader("Aspek / Topik")
                st.info(f"**{asp_label}**")
                st.metric("Confidence", f"{asp_conf:.2%}")

with tab2:
    st.header("Upload File CSV/Excel")
    uploaded_file = st.file_uploader("Upload file Anda (pastikan ada kolom teks)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("Preview Data:")
            st.dataframe(df.head())
            
            text_col = st.selectbox("Pilih kolom yang berisi ulasan/text:", df.columns)
            
            if st.button("Proses Prediksi Batch"):
                with st.spinner("Sedang memproses..."):
                    # 1. Preprocessing
                    # Convert to string to handle potential non-string inputs safely
                    texts = df[text_col].astype(str).tolist()
                    
                    # Sentiment Batch
                    processed_sent = [preprocess_sentiment(t) for t in texts]
                    seq_sent = tok_sent.texts_to_sequences(processed_sent)
                    pad_sent = pad_sequences(seq_sent, maxlen=MAX_LEN, padding="post", truncating="post")
                    
                    sent_probs = sentiment_model.predict(pad_sent, verbose=0)
                    
                    sent_labels = ["Positif" if p[0] >= 0.5 else "Negatif" for p in sent_probs]
                    sent_confs  = [p[0] if p[0] >= 0.5 else 1 - p[0] for p in sent_probs]
                    
                    # Topic Batch
                    processed_topic = [preprocess_topic(t) for t in texts]
                    seq_topic = tok_asp.texts_to_sequences(processed_topic)
                    pad_topic = pad_sequences(seq_topic, maxlen=MAX_LEN, padding="post", truncating="post")
                    
                    topic_probs = aspect_model.predict(pad_topic, verbose=0)
                    
                    topic_indices = np.argmax(topic_probs, axis=1)
                    topic_confs   = np.max(topic_probs, axis=1)
                    topic_labels  = [TOPIC_MAPPING.get(idx, f"Topic {idx}") for idx in topic_indices]
                    
                    # Store to DF
                    df['Prediksi Sentimen'] = sent_labels
                    df['Conf Sentimen'] = sent_confs
                    
                    df['Prediksi Aspek'] = topic_labels
                    df['Conf Aspek'] = topic_confs
                    
                    st.success("Proses selesai!")
                    st.dataframe(df)
                    
                    # CSV Download
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Hasil sebagai CSV",
                        data=csv_data,
                        file_name="hasil_prediksi_absa.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
