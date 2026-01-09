import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import numpy as np
# Uncomment these if you want to use Sastrawi for preprocessing
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Set page config
st.set_page_config(
    page_title="ABSA - Stockbit Analysis",
    page_icon="📊",
    layout="centered"
)

# Constants
MAX_SEQUENCE_LENGTH = 100
TOPIC_MAPPING = {
    0: 'Desain & Pengalaman Pengguna (UI/UX)',  # User-friendly, Tampilan, CS
    1: 'Kemudahan Penggunaan & Pemula',         # Mudah, Simple, Cocok untuk Pemula
    2: 'Kepuasan & Kepercayaan Pengguna',       # Puas, Terpercaya, Mantap
    3: 'Fitur Trading & Edukasi Investasi',     # Chart, Analisa, Belajar Saham
    4: 'Kinerja Sistem & Fitur Order',          # Trailing Stop, Market Hours, Lag
    5: 'Administrasi Akun & RDN',               # Bank Jago, RDN, Pendaftaran
    6: 'Transaksi Keuangan (Deposit/WD)',       # Deposit, Withdraw, Dividen
    7: 'Stabilitas Aplikasi & Maintenance'      # Error, Crash, Server Down
}

# TOPIC_MAPPING = {
#     0: 'Akses Aplikasi',
#     1: 'Fitur & Kemudahan Investasi',
#     2: 'Fitur & Kemudahan Investasi',
#     3: 'Fitur & Kemudahan Investasi',
#     4: 'Stabilitas Sistem Perdagangan',
#     5: 'Integrasi Rekening Bank',
#     6: 'Proses Transaksi Keuangan',
#     7: 'Stabilitas Sistem Perdagangan'
# }

@st.cache_resource
def load_resources():
    # Load Models
    try:
        sentiment_model = load_model('sentiment_model.h5')
        topic_model = load_model('topic_model.h5')
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

    # Load Tokenizers
    # Note: You must export your tokenizers from the notebook as pickles!
    try:
        with open('tokenizer_sentiment.pkl', 'rb') as f:
            sentiment_tokenizer = pickle.load(f)
        with open('tokenizer_topic.pkl', 'rb') as f:
            topic_tokenizer = pickle.load(f)
    except FileNotFoundError:
        st.warning("Tokenizer files (tokenizer_sentiment.pkl, tokenizer_topic.pkl) not found. Text processing will fail until these are provided.")
        return sentiment_model, topic_model, None, None
    except Exception as e:
        st.error(f"Error loading tokenizers: {e}")
        return sentiment_model, topic_model, None, None

    return sentiment_model, topic_model, sentiment_tokenizer, topic_tokenizer

def load_stopwords():
    stopwords_sentiment = None
    stopwords_topic = None
    
    # Load for Sentiment (Filtered)
    try:
        with open('stopwords-id-filtered.txt', 'r') as f:
            stopwords_sentiment = set(line.strip() for line in f)
    except FileNotFoundError:
        st.warning("Stopwords file (stopwords-id-filtered.txt) not found.")

    # Load for Topic (Full)
    try:
        with open('stopwords-id.txt', 'r') as f:
            stopwords_topic = set(line.strip() for line in f)
    except FileNotFoundError:
        st.warning("Stopwords file (stopwords-id.txt) not found.")
        
    return stopwords_sentiment, stopwords_topic

def preprocess_base(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"&\w+;", " ", text)             # hapus HTML entities (&amp;, &quot;, dll.)
    text = re.sub(r"[^a-z]", " ", text)         # hapus semua karakter kecuali a-z
    text = re.sub(r"\t", " ", text)             # ganti tab dengan spasi
    text = re.sub(r"\n", " ", text)             # ganti new line dengan spasi
    text = re.sub(r"\s+", " ", text)            # ganti spasi > 1 dengan 1 spasi
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # ganti huruf berulang ≥ 3 jadi 2
    text = text.strip()
    return text

def preprocess_for_sentiment(text, stopwords_set):
    text = preprocess_base(text)
    
    # Stopword Removal (Filtered)
    if stopwords_set:
        text = ' '.join([word for word in text.split() if word not in stopwords_set])
            
    return text

def preprocess_for_topic(text, stopwords_set):
    text = preprocess_base(text)
    
    # Stopword Removal (Full)
    if stopwords_set:
        text = ' '.join([word for word in text.split() if word not in stopwords_set])
    
    # Stemming
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        text = stemmer.stem(text)
    except ImportError:
        pass
        
    return text

def main():
    st.title("📊 Aspect Based Sentiment Analysis")
    st.markdown("Analisis Sentimen dan Topik untuk Ulasan Aplikasi Stockbit")
    
    # Load resources
    sentiment_model, topic_model, sent_tokenizer, topic_tokenizer = load_resources()
    stopwords_sentiment, stopwords_topic = load_stopwords()
    
    if not sentiment_model or not topic_model:
        st.warning("Models could not be loaded. Please check file paths.")
        return

    # Input section
    with st.form("analysis_form"):
        user_input = st.text_area("Masukkan Ulasan Anda:", height=100, placeholder="Contoh: Aplikasi ini sangat membantu untuk investasi saham bagi pemula.")
        submitted = st.form_submit_button("Analisis")

    if submitted and user_input:
        if sent_tokenizer is None or topic_tokenizer is None:
            st.error("Tokenizers are missing. Please upload 'tokenizer_sentiment.pkl' and 'tokenizer_topic.pkl'.")
        else:
            # Preprocess
            processed_sent = preprocess_for_sentiment(user_input, stopwords_sentiment)
            processed_topic = preprocess_for_topic(user_input, stopwords_topic)
            
            # 1. Sentiment Analysis
            sent_seq = sent_tokenizer.texts_to_sequences([processed_sent])
            sent_pad = pad_sequences(sent_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
            
            sent_prob = sentiment_model.predict(sent_pad)[0][0]
            sentiment_label = "Positif" if sent_prob > 0.5 else "Negatif"
            sentiment_color = "green" if sent_prob > 0.5 else "red"
            
            prob_pos = float(sent_prob)
            prob_neg = 1.0 - prob_pos
            
            # 2. Topic Analysis
            topic_seq = topic_tokenizer.texts_to_sequences([processed_topic])
            topic_pad = pad_sequences(topic_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
            
            topic_probs = topic_model.predict(topic_pad)[0]
            topic_idx = np.argmax(topic_probs)
            topic_label = TOPIC_MAPPING.get(topic_idx, "Unknown")
            topic_confidence = topic_probs[topic_idx]

            # Display Results
            st.write("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentimen")
                st.markdown(f"<h3 style='color: {sentiment_color};'>{sentiment_label}</h3>", unsafe_allow_html=True)
                
                # Menampilkan probabilitas detail
                sub_col_a, sub_col_b = st.columns(2)
                with sub_col_a:
                    st.metric("Positif", f"{prob_pos:.1%}")
                with sub_col_b:
                    st.metric("Negatif", f"{prob_neg:.1%}")
                
                # Progress bar visual (semakin penuh semakin positif)
                # st.caption("Skor Probabilitas:")
                # st.progress(prob_pos)

            with col2:
                st.subheader("Aspek / Topik")
                st.info(f"**{topic_label}**")
                st.metric("Confidence", f"{topic_confidence:.1%}")
            
            with st.expander("Lihat Detail Preprocessing"):
                st.write("**Original Text:**")
                st.write(user_input)
                st.write(f"**Processed for Sentiment:** (Stopwords: {len(stopwords_sentiment) if stopwords_sentiment else 0} words loaded)")
                st.write(processed_sent)
                st.write(f"**Processed for Topic:** (Stopwords: {len(stopwords_topic) if stopwords_topic else 0} words loaded + Stemming)")
                st.write(processed_topic)

if __name__ == "__main__":
    main()
