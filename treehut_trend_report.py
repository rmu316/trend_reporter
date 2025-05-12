# treehut_trend_report.py

import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from textblob import TextBlob
from wordcloud import WordCloud
from datetime import datetime
from collections import defaultdict
import os

# --------- Step 1: Load and Preprocess Data --------- #
def load_and_clean_data(filepath):
    # Load CSV, clean missing timestamps/comments, normalize date, clean text, compute sentiment
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["comment_text"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=["timestamp"])
    df['date'] = df['timestamp'].dt.date
    # Normalize comment text and remove common and brand-specific stopwords
    df['clean_comment'] = df['comment_text'].apply(clean_comment)
    df['sentiment'] = df['comment_text'].apply(get_sentiment)
    return df

def clean_comment(text):
    stopwords = set(["i","me","my","myself","we","our","you","your","yours","he","him","his","she",
                     "her","they","them","their","what","which","who","this","that","am","is","are",
                     "was","were","be","been","being","have","has","had","having","do","does","did",
                     "doing","a","an","the","and","but","if","or","because","as","until","while",
                     "of","at","by","for","with","about","against","between","into","through","during",
                     "before","after","above","below","to","from","up","down","in","out","on","off",
                     "over","under","again","further","then","once","here","there","when","where","why",
                     "how","all","any","both","each","few","more","most","other","some","such","no",
                     "nor","not","only","own","same","so","than","too","very","can","will","just",
                     "tree", "hut", "treehut", "pr", "love", "please", "it", "omg", "yes", "pleaseee", "treehutpr",
                     "good", "would", "want", "coco"])
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    return ' '.join(tokens)

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# --------- Step 2: Topic Modeling --------- #
def extract_topics(df, n_topics=10):
    # Apply TF-IDF and NMF topic modeling, then assign a unique label to each topic
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2), min_df=2, max_df=0.95)
    tfidf_matrix = tfidf.fit_transform(df['clean_comment'])

    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(tfidf_matrix)
    H = nmf.components_

    feature_names = tfidf.get_feature_names_out()
    topics = []
    topic_labels = []
    used_keywords = set()

    label_map = {
        "trent": "Name Mentions",
        "april": "April Fools",
        "favorite": "Product Favorites",
        "scent": "Fragrance",
        "need": "Wishlist",
        "try": "Product Curiosity",
        "brand": "Brand Sentiment",
        "treehutpr": "PR Requests",
        "pr": "PR Requests",
        "want": "Desire / Interest",
        "good": "Positive Feedback",
        "would": "Hypothetical Wants",
        "coco": "Coconut Product Mentions"
    }

    for topic_idx, topic in enumerate(H):
        sorted_indices = topic.argsort()[::-1]
        label = None
        for idx in sorted_indices:
            keyword = feature_names[idx]
            if keyword not in used_keywords:
                used_keywords.add(keyword)
                label = label_map.get(keyword, keyword)
                break
        topics.append("Topic #{}: {}".format(topic_idx + 1, label))
        topic_labels.append(label)

    df['topic'] = W.argmax(axis=1)
    df['topic_label'] = df['topic'].apply(lambda x: topic_labels[x])

    topic_df = pd.DataFrame({"Topic #": [f"T{i+1}" for i in range(n_topics)], "Top Word": topic_labels})
    topic_df.to_csv("topic_summary.csv", index=False)

    generate_wordclouds(df, n_topics)
    generate_media_engagement_report(df)

    return df, topics

# --------- Step 3: Word Cloud Generation --------- #
def generate_wordclouds(df, n_topics):
    # Generate and save word cloud images for each topic using cleaned comment text
    os.makedirs("wordclouds", exist_ok=True)
    for topic_id in range(n_topics):
        text = " ".join(df[df['topic'] == topic_id]['clean_comment'])
        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            wordcloud.to_file(f"wordclouds/topic_{topic_id+1}.png")

# --------- Step 4: Media-Level Engagement Report --------- #
def generate_media_engagement_report(df):
    media_stats = df.groupby(['media_id', 'media_caption']).agg(
        comment_count=('comment_text', 'count'),
        avg_sentiment=('sentiment', 'mean')
    ).reset_index().sort_values(by='comment_count', ascending=False)

    media_stats.to_csv("media_engagement_summary.csv", index=False)

# --------- Step 5: Visualization --------- #
def plot_topic_trends(df, topics):
    topic_counts_by_day = df.groupby(['date', 'topic']).size().unstack(fill_value=0)
    topic_counts_by_day.plot(kind='line', figsize=(12, 6))
    plt.title("Topic Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Comments")
    plt.legend(["T{}".format(i+1) for i in range(len(topics))], title="Topics")
    plt.tight_layout()
    plt.savefig("topic_trends.png")
    plt.close()

def plot_sentiment_trend(df):
    sentiment_by_day = df.groupby('date')['sentiment'].mean()
    sentiment_by_day.plot(kind='line', figsize=(12, 5), color='green')
    plt.title("Average Sentiment Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment (TextBlob Polarity)")
    plt.axhline(0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig("sentiment_trend.png")
    plt.close()

# --------- Step 6: Main --------- #
def main():
    # Main execution: runs full pipeline from loading data to exporting processed results and charts
    filepath = "engagements.csv"
    df = load_and_clean_data(filepath)
    df, topics = extract_topics(df, n_topics=5)

    print("Extracted Topics:")
    for t in topics:
        print(t)

    plot_topic_trends(df, topics)
    plot_sentiment_trend(df)

    df.to_csv("processed_engagements.csv", index=False)

    print("Charts saved: 'topic_trends.png', 'sentiment_trend.png'")
    print("Word clouds saved in 'wordclouds/' directory.")
    print("Topic summary saved as 'topic_summary.csv'.")
    print("Media engagement summary saved as 'media_engagement_summary.csv'.")
    print("Full processed dataset saved as 'processed_engagements.csv'.")

if __name__ == "__main__":
    main()
