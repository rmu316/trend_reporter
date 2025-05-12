# treehut_dashboard.py

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load preprocessed data
@st.cache_data

def load_data():
    df = pd.read_csv("processed_engagements.csv", parse_dates=['timestamp'], on_bad_lines='skip')
    if 'clean_comment' in df.columns and 'sentiment' in df.columns and 'topic_label' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        return df
    else:
        st.error("Required columns missing. Please preprocess data first.")
        return pd.DataFrame()

# Main app
st.set_page_config(page_title="Tree Hut Comment Trends", layout="wide")
st.title("🌿 Tree Hut Instagram Comment Insights Dashboard")

df = load_data()

if not df.empty:
    # Sidebar filters
    st.sidebar.header("Filters")
    unique_topics = df['topic_label'].unique()
    selected_topics = st.sidebar.multiselect("Choose Topics", unique_topics, default=unique_topics)
    date_range = st.sidebar.date_input("Date Range", [df['date'].min(), df['date'].max()])

    filtered_df = df[(df['topic_label'].isin(selected_topics)) &
                     (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]

    # Topic trend
    st.subheader("📈 Topic Frequency Over Time")
    topic_counts = filtered_df.groupby(['date', 'topic_label']).size().unstack().fillna(0)
    st.line_chart(topic_counts)

    # Sentiment trend
    st.subheader("😊 Average Sentiment Over Time")
    sentiment_by_day = filtered_df.groupby('date')['sentiment'].mean()
    st.line_chart(sentiment_by_day)

    # Word cloud
    st.subheader("☁️ Word Cloud for Selected Topics")
    text = " ".join(filtered_df['clean_comment'].dropna())
    if text:
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("No text available to generate word cloud.")

    # Media-level table
    st.subheader("📋 Media Engagement Summary")
    if 'media_id' in df.columns:
        media_stats = df.groupby(['media_id', 'media_caption']).agg(
            comment_count=('comment_text', 'count'),
            avg_sentiment=('sentiment', 'mean')
        ).reset_index().sort_values(by='comment_count', ascending=False)
        st.dataframe(media_stats.head(20))

        # Add CSV download
        st.download_button(
            label="📥 Download Media Engagement Summary",
            data=media_stats.to_csv(index=False),
            file_name="media_engagement_summary.csv",
            mime="text/csv"
        )
    else:
        st.warning("Media data not available in this CSV.")

    # Export full filtered dataset
    st.subheader("📤 Export Filtered Comments")
    st.download_button(
        label="📥 Download Filtered Comments",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_comments.csv",
        mime="text/csv"
    )
