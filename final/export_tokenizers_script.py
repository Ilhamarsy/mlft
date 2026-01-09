import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

# INSTRUCTIONS:
# Copy and run this code in your Google Colab / Notebook environment
# where you have trained the models and have the 'tokenizer' objects in memory.

# Assuming 'tokenizer' is the one used for Sentiment Analysis (fitted on X_train from mlft_preprocessedv2.csv)
# and 'tokenizer_merged' (or whatever variable name used) for Topic Analysis.
# If you used the SAME tokenizer for both, save it twice or modify the loading code.

# Based on the notebook, there were two distinct tokenizers created:
# 1. cell 7: tokenizer (Sentiment)
# 2. cell 31: tokenizer (Topic) - Variable name was overwritten as 'tokenizer' in cell 31!

# CRITICAL: Since the variable 'tokenizer' was overwritten in cell 31, 
# you might need to re-run the cells to regenerate the specific tokenizer for each task 
# before saving.

# Step 1: Re-create/Save Sentiment Tokenizer
print("Generating/Saving Sentiment Tokenizer...")
# re-run text loading for sentiment
df_sent = pd.read_csv('mlft_preprocessedv2.csv')
X_sent = df_sent['stopwords'].astype(str)
# ... split ...
tokenizer_sent = Tokenizer(oov_token="<OOV>")
tokenizer_sent.fit_on_texts(X_sent) # Or X_train from sentiment split

with open('tokenizer_sentiment.pkl', 'wb') as handle:
    pickle.dump(tokenizer_sent, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved tokenizer_sentiment.pkl")

# Step 2: Re-create/Save Topic Tokenizer
print("Generating/Saving Topic Tokenizer...")
df_topic = pd.read_csv('mlft_preprocessed_topic.csv')
X_topic = df_topic['stem'].astype(str)
# ... split ...
tokenizer_topic = Tokenizer(oov_token="<OOV>")
tokenizer_topic.fit_on_texts(X_topic) # Or X_train from topic split

with open('tokenizer_topic.pkl', 'wb') as handle:
    pickle.dump(tokenizer_topic, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved tokenizer_topic.pkl")

print("Please download these two .pkl files and place them in the same folder as app.py")
