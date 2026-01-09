import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import hdbscan
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('hasil_preprocessing_hdbscan_embedding.csv')

# Parse embeddings - they're stored as string representations of arrays
def parse_embedding(embedding_str):
    # Remove brackets and split by whitespace, then convert to float
    if pd.isna(embedding_str):
        return np.array([])
    try:
        # Remove brackets and split
        embedding_str = str(embedding_str).strip('[]')
        if embedding_str == '':
            return np.array([])
        # Split by whitespace and convert to float
        values = [float(x) for x in embedding_str.split()]
        return np.array(values)
    except:
        return np.array([])

print("Parsing embeddings...")
# Apply parsing function to embeddings
embeddings = df['embedding'].apply(parse_embedding)

# Filter out any empty embeddings
valid_mask = embeddings.apply(len) > 0
embeddings = embeddings[valid_mask]
df_clean = df[valid_mask].copy()

# Convert to numpy array
X = np.vstack(embeddings.values)

print(f"Dataset shape: {X.shape}")
print(f"Number of documents: {len(df_clean)}")

# Use the best parameter set found from testing
print("\n" + "="*50)
print("PERFORMING FINAL HDBSCAN CLUSTERING")
print("="*50)
print("Using optimal parameters: min_cluster_size=20, min_samples=1, metric='euclidean'")

# HDBSCAN Clustering with optimal parameters
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,
    min_samples=1,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

cluster_labels = clusterer.fit_predict(X)

# Add cluster labels to dataframe
df_clean['cluster'] = cluster_labels

# Analyze clustering results
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"\nClustering Results:")
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"Percentage of noise: {(n_noise/len(cluster_labels))*100:.1f}%")

# Cluster statistics
cluster_stats = {}
for cluster_id in set(cluster_labels):
    if cluster_id != -1:  # Exclude noise
        cluster_size = (cluster_labels == cluster_id).sum()
        cluster_stats[cluster_id] = cluster_size

print(f"\nCluster sizes:")
for cluster_id, size in sorted(cluster_stats.items()):
    print(f"Cluster {cluster_id}: {size} documents")

# Topic extraction from clusters
def extract_topics_from_cluster(cluster_id, top_n=10):
    """Extract most frequent words from a cluster"""
    cluster_data = df_clean[df_clean['cluster'] == cluster_id]
    all_words = []

    # Extract words from clean_content
    for content in cluster_data['clean_content']:
        if pd.notna(content):
            # Simple tokenization - split by whitespace and remove punctuation
            words = re.findall(r'\b\w+\b', str(content).lower())
            # Filter out very common and very short words
            words = [word for word in words if len(word) > 2 and word not in ['yang', 'dan', 'di', 'untuk', 'dengan', 'ada', 'ini', 'itu', 'atau', 'bisa', 'tapi', 'saya', ' aplikasi']]
            all_words.extend(words)

    # Count word frequencies
    word_freq = Counter(all_words)
    return word_freq.most_common(top_n)

print(f"\n" + "="*50)
print("TOPIC ANALYSIS")
print("="*50)

# Extract topics for each cluster
cluster_topics = {}
for cluster_id in sorted(cluster_stats.keys()):
    top_words = extract_topics_from_cluster(cluster_id, top_n=10)
    cluster_topics[cluster_id] = top_words

    print(f"\nCluster {cluster_id} ({cluster_stats[cluster_id]} documents):")
    print("Top words:", [word for word, freq in top_words[:5]])
    print("Full top 10:", top_words)

# Visualization
print(f"\nCreating visualizations...")

# 1. t-SNE visualization
print("Creating t-SNE visualization...")
# Limit t-SNE to reasonable sample size for performance
sample_size = min(1000, len(X))
sample_indices = np.random.choice(len(X), sample_size, replace=False)
X_sample = X[sample_indices]
labels_sample = cluster_labels[sample_indices]

tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_sample)//4))
X_tsne = tsne.fit_transform(X_sample)

# Create t-SNE plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_sample, cmap='tab10', s=50, alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('HDBSCAN Clustering Results (t-SNE Visualization)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig('clustering_results_tsne_final.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. PCA visualization
print("Creating PCA visualization...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', s=50, alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.title('HDBSCAN Clustering Results (PCA Visualization)')
plt.xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.tight_layout()
plt.savefig('clustering_results_pca_final.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Cluster size distribution
plt.figure(figsize=(12, 6))
cluster_sizes = [cluster_stats[i] for i in sorted(cluster_stats.keys())]
cluster_ids = [f'Cluster {i}' for i in sorted(cluster_stats.keys())]

plt.bar(cluster_ids, cluster_sizes)
plt.title('Cluster Size Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Documents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('cluster_size_distribution_final.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Create word clouds for each cluster
print("Creating word clouds...")
for cluster_id in sorted(cluster_stats.keys()):
    top_words = extract_topics_from_cluster(cluster_id, top_n=30)
    if top_words:
        # Create word frequency dictionary for word cloud
        word_freq_dict = {word: freq for word, freq in top_words}

        plt.figure(figsize=(10, 6))
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=30,
            colormap='viridis'
        ).generate_from_frequencies(word_freq_dict)

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Cluster {cluster_id} ({cluster_stats[cluster_id]} documents)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'wordcloud_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
        plt.show()

# Save results
print(f"\nSaving results...")

# Save dataframe with cluster labels
df_clean.to_csv('hasil_clustering_hdbscan_final.csv', index=False)
print("Saved clustered data to 'hasil_clustering_hdbscan_final.csv'")

# Save topic analysis
topics_df = pd.DataFrame([
    {
        'cluster': cluster_id,
        'size': cluster_stats[cluster_id],
        'top_words': ', '.join([word for word, freq in topic_words[:5]]),
        'full_topic': str(topic_words)
    }
    for cluster_id, topic_words in cluster_topics.items()
])

topics_df.to_csv('cluster_topics_hdbscan_final.csv', index=False)
print("Saved topic analysis to 'cluster_topics_hdbscan_final.csv'")

# Create summary analysis
summary_data = []
for cluster_id in sorted(cluster_stats.keys()):
    cluster_docs = df_clean[df_clean['cluster'] == cluster_id]

    # Calculate average sentiment
    avg_sentiment = cluster_docs['score'].mean()

    # Get example documents
    examples = cluster_docs['content'].head(3).tolist()

    top_words = [word for word, freq in cluster_topics[cluster_id][:5]]

    summary_data.append({
        'cluster_id': cluster_id,
        'size': cluster_stats[cluster_id],
        'percentage': (cluster_stats[cluster_id] / len(df_clean)) * 100,
        'avg_sentiment': avg_sentiment,
        'top_words': ', '.join(top_words),
        'example_1': examples[0][:100] + '...' if len(examples) > 0 else '',
        'example_2': examples[1][:100] + '...' if len(examples) > 1 else '',
        'example_3': examples[2][:100] + '...' if len(examples) > 2 else ''
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('clustering_summary_final.csv', index=False)
print("Saved clustering summary to 'clustering_summary_final.csv'")

# Summary statistics
print(f"\n" + "="*60)
print("FINAL CLUSTERING SUMMARY")
print("="*60)
print(f"Total documents processed: {len(df_clean)}")
print(f"Embedding dimension: {X.shape[1]}")
print(f"HDBSCAN parameters: min_cluster_size=20, min_samples=1, metric='euclidean'")
print(f"Number of topics found: {n_clusters}")
print(f"Noise documents: {n_noise} ({(n_noise/len(cluster_labels))*100:.1f}%)")

if n_clusters > 0:
    print(f"\nAverage cluster size: {np.mean(list(cluster_stats.values())):.1f}")
    print(f"Largest cluster size: {max(cluster_stats.values())}")
    print(f"Smallest cluster size: {min(cluster_stats.values())}")

    print(f"\n" + "="*60)
    print("DETAILED TOPIC SUMMARY")
    print("="*60)

    for i, row in summary_df.iterrows():
        print(f"\nTOPIC {row['cluster_id']} ({row['size']} documents, {row['percentage']:.1f}%):")
        print(f"  Sentiment: {'Positive' if row['avg_sentiment'] > 3.5 else 'Negative' if row['avg_sentiment'] < 2.5 else 'Neutral'} ({row['avg_sentiment']:.2f})")
        print(f"  Top words: {row['top_words']}")
        print(f"  Example: {row['example_1']}")

print(f"\nFiles generated:")
print(f"- clustering_results_tsne_final.png")
print(f"- clustering_results_pca_final.png")
print(f"- cluster_size_distribution_final.png")
print(f"- wordcloud_cluster_X.png (one for each cluster)")
print(f"- hasil_clustering_hdbscan_final.csv")
print(f"- cluster_topics_hdbscan_final.csv")
print(f"- clustering_summary_final.csv")

print(f"\nTopic clustering analysis completed successfully!")
print(f"Found {n_clusters} distinct topics from {len(df_clean)} documents.")