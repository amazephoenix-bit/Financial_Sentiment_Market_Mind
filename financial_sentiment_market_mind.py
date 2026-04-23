import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib
import os
import feedparser
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from streamlit_autorefresh import st_autorefresh

# 1. Streamlit Page Configuration (MUST be first)
st.set_page_config(page_title="Market-Mind: Financial Sentiment Dashboard", layout="wide")

# 2. Setup NLTK and Spacy
@st.cache_resource
def load_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_resources()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 3. Text Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# 4. Model & Vectorizer (Cached)
@st.cache_resource
def get_trained_model():
    if os.path.exists("data.csv"):
        df = pd.read_csv("data.csv")
        df['clean_text'] = df['Sentence'].apply(clean_text)
        
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['clean_text'])
        y = df['Sentiment']
        
        model = LogisticRegression()
        model.fit(X, y)
        return model, vectorizer
    else:
        st.error("data.csv not found. Please upload the dataset to the repository.")
        return None, None

model, tfidf = get_trained_model()

# 5. Sentiment Analysis Functions
def predict_sentiment(text):
    if model and tfidf:
        text_clean = clean_text(text)
        vector = tfidf.transform([text_clean]).toarray()
        prediction = model.predict(vector)
        return prediction[0]
    return "Unknown"

def extract_companies(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]

company_to_ticker = {
    "Tesla": "TSLA", "Apple": "AAPL", "Microsoft": "MSFT", 
    "Amazon": "AMZN", "Google": "GOOGL", "Nvidia": "NVDA"
}

def analyze_news(text):
    companies = extract_companies(text)
    sentiment = predict_sentiment(text)
    results = []
    for company in companies:
        ticker = company_to_ticker.get(company, "UNKNOWN")
        results.append({"company": company, "ticker": ticker, "sentiment": sentiment})
    return results

# 6. UI Logic
st.title("📈 Market-Mind: Financial Sentiment Dashboard")
st.sidebar.header("Dashboard Settings")

# Auto-refresh every 5 minutes
st_autorefresh(interval=300000, key="datarefresh")

# Fetch Live RSS Feed
@st.cache_data(ttl=600)
def fetch_live_news():
    url = "https://finance.yahoo.com/rss/topstories"
    feed = feedparser.parse(url)
    results = []
    for entry in feed.entries[:15]:
        headline = entry.title
        analysis = analyze_news(headline)
        if not analysis: # If no specific company found, still show sentiment
            results.append({"headline": headline, "company": "General", "ticker": "N/A", "sentiment": predict_sentiment(headline)})
        for item in analysis:
            item["headline"] = headline
            results.append(item)
    return pd.DataFrame(results)

news_df = fetch_live_news()

# 7. Metrics & Visuals
if not news_df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("📰 Headlines", len(news_df))
    col2.metric("🟢 Bullish", len(news_df[news_df['sentiment'] == 'positive']))
    col3.metric("🔴 Bearish", len(news_df[news_df['sentiment'] == 'negative']))

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sentiment by Ticker")
        summary = news_df.groupby("ticker")["sentiment"].value_counts().unstack().fillna(0)
        st.bar_chart(summary)
    
    with c2:
        st.subheader("Sentiment Distribution")
        st.bar_chart(news_df['sentiment'].value_counts())

    st.divider()
    st.subheader("📰 Latest Financial News Analysis")
    
    # Filter by Stock
    selected_ticker = st.sidebar.selectbox("Filter by Stock", options=["All"] + list(news_df['ticker'].unique()))
    if selected_ticker != "All":
        news_df = news_df[news_df['ticker'] == selected_ticker]
    
    st.dataframe(news_df, use_container_width=True)
else:
    st.write("No news data found.")
