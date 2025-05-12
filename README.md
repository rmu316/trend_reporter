# 🌿 Tree Hut Instagram Insights Dashboard

This project analyzes ~18,000 Instagram comments from Tree Hut's March 2025 content to identify trends, sentiment, and user interests. It includes a data pipeline and an interactive dashboard built with Streamlit for social media teams to explore insights in real time.

---

## 📊 Features

- **Topic Modeling**: Automatically detects 5 key themes in user comments using TF-IDF and NMF
- **Sentiment Analysis**: Tracks emotional response to posts over time
- **Word Clouds**: Visual summaries of dominant terms by topic
- **Media-Level Engagement**: Shows which posts had the most comments and strongest reactions
- **Interactive Dashboard**: Filter comments by topic and date with instant visual updates

---

## 📁 Project Contents

| File / Folder                  | Description |
|-------------------------------|-------------|
| `treehut_trend_report.py`     | Main script to preprocess and analyze the data |
| `treehut_dashboard.py`        | Interactive Streamlit dashboard |
| `requirements.txt`            | Required Python libraries |
| `README.md`                   | This documentation file |

---

## 🚀 Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the data pipeline
python3 treehut_trend_report.py

# Start the dashboard
streamlit run treehut_dashboard.py
```

---

## 🌐 Deployment

You can deploy this dashboard using [Streamlit Cloud](https://streamlit.io/cloud). Push your repo to GitHub, then link it on Streamlit Cloud with `treehut_dashboard.py` as the entry point.

---

## 🤖 Tech Stack

- Python (pandas, matplotlib, scikit-learn, textblob)
- Streamlit (interactive UI)
- NMF + TF-IDF for topic modeling
- WordCloud, line charts, and CSV exports

---

## 📬 Contact

For questions or feedback, feel free to reach out or open an issue in the GitHub repo.