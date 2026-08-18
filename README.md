# 📈 Market-Mind: Financial Sentiment Dashboard

**Market-Mind** is a machine-learning powered financial sentiment analysis dashboard built with **Python and Streamlit**. It analyzes live financial news headlines, classifies their sentiment as **positive or negative**, identifies mentioned companies, maps supported companies to stock tickers, and presents the results through an interactive dashboard.

The project combines **Natural Language Processing (NLP)**, **TF-IDF text representation**, **Logistic Regression**, **Named Entity Recognition (NER)**, and live financial news feeds to provide a simple market-sentiment monitoring tool.

> ⚠️ **Disclaimer:** This project is intended for educational and research purposes only. It does not provide financial advice, investment recommendations, or guaranteed market predictions.

---

## 🚀 Features

* 📰 Fetches the latest financial headlines from Yahoo Finance RSS
* 🤖 Performs automated financial sentiment classification
* 🧹 Cleans and preprocesses text using NLTK
* 🔤 Converts text into numerical features using **TF-IDF**
* 📊 Uses **Logistic Regression** for sentiment classification
* 🏢 Detects organizations and companies using **spaCy NER**
* 💹 Maps supported companies to stock tickers
* 📈 Displays sentiment distribution through interactive charts
* 🔎 Allows filtering news by stock ticker
* 🔄 Automatically refreshes market news every 5 minutes
* ⚡ Uses Streamlit caching to improve application performance

---

## 🧠 How It Works

The application follows a simple NLP pipeline:

```text
                    Financial News
                          │
                          ▼
                 Yahoo Finance RSS
                          │
                          ▼
                  Text Preprocessing
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       NLTK Cleaning              spaCy NER
             │                         │
             ▼                         ▼
        Lemmatization             Companies
             │                         │
             └────────────┬────────────┘
                          ▼
                    TF-IDF Vectorizer
                          │
                          ▼
                 Logistic Regression
                          │
                          ▼
                  Sentiment Prediction
                          │
                          ▼
                 Streamlit Dashboard
```

### 1. Text Preprocessing

News headlines are:

* Converted to lowercase
* Stripped of non-alphabetic characters
* Tokenized
* Filtered using English stopwords
* Lemmatized using WordNet

### 2. Feature Extraction

The cleaned text is transformed into numerical features using:

**TF-IDF (Term Frequency–Inverse Document Frequency)**

The vectorizer is configured with a maximum of **5,000 features**.

### 3. Sentiment Classification

A **Logistic Regression** classifier is trained on the provided financial sentiment dataset.

The model predicts the sentiment of each incoming financial headline.

### 4. Company Detection

spaCy's `en_core_web_sm` model is used for **Named Entity Recognition** to identify organizations mentioned in headlines.

Currently supported ticker mappings include:

| Company   | Ticker |
| --------- | ------ |
| Tesla     | TSLA   |
| Apple     | AAPL   |
| Microsoft | MSFT   |
| Amazon    | AMZN   |
| Google    | GOOGL  |
| Nvidia    | NVDA   |

Companies not included in the mapping are labelled as `UNKNOWN`.

### 5. Live News Analysis

The application retrieves the latest headlines from Yahoo Finance's financial RSS feed and analyzes each headline automatically.

The dashboard refreshes the data every **5 minutes**.

---

## 📊 Dashboard

The Streamlit dashboard provides:

### Key Metrics

* 📰 Total analyzed headlines
* 🟢 Number of positive headlines
* 🔴 Number of negative headlines

### Visualizations

#### Sentiment by Ticker

Displays the number of positive and negative headlines associated with each ticker.

#### Sentiment Distribution

Shows the overall distribution of sentiment across the collected headlines.

### News Table

The dashboard displays:

* Headline
* Company
* Stock ticker
* Sentiment

A sidebar filter allows users to view news associated with a particular stock.

---

## 🛠️ Tech Stack

| Technology            | Purpose                     |
| --------------------- | --------------------------- |
| Python                | Core programming language   |
| Streamlit             | Interactive dashboard       |
| Pandas                | Data manipulation           |
| Scikit-learn          | Machine learning            |
| NLTK                  | Text preprocessing          |
| spaCy                 | Named Entity Recognition    |
| TF-IDF                | Text feature extraction     |
| Logistic Regression   | Sentiment classification    |
| Feedparser            | Financial RSS feed parsing  |
| Streamlit Autorefresh | Automatic dashboard updates |

---

## 📁 Project Structure

```text
Financial_Sentiment_Market_Mind/
│
├── data.csv
│   └── Financial sentiment training dataset
│
├── financial_sentiment_market_mind.py
│   └── Main Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
└── README.md
    └── Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/amazephoenix-bit/Financial_Sentiment_Market_Mind.git
```

### 2. Navigate into the project

```bash
cd Financial_Sentiment_Market_Mind
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

The project dependencies include Streamlit, Pandas, Scikit-learn, NLTK, spaCy, Feedparser, and Streamlit Autorefresh.

---

## ▶️ Running the Application

Start the Streamlit application with:

```bash
streamlit run financial_sentiment_market_mind.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser to access the dashboard.

---

## 📚 Dataset

The project uses `data.csv` as the training dataset.

The application expects the dataset to contain:

```text
Sentence
Sentiment
```

The `Sentence` column contains financial text, while `Sentiment` contains the corresponding sentiment label.

During startup, the application cleans the sentences, creates TF-IDF features, and trains the Logistic Regression model.

---

## 🔬 Machine Learning Pipeline

The model pipeline can be summarized as:

```text
Raw Financial Text
        ↓
Text Cleaning
        ↓
Stopword Removal
        ↓
Lemmatization
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression
        ↓
Sentiment
```

This project demonstrates how a traditional NLP + machine-learning pipeline can be integrated into a real-time data application.

---

## 🎯 Project Goals

The main goals of Market-Mind are to:

1. Apply NLP techniques to financial text.
2. Build a practical sentiment classification system.
3. Integrate machine learning with live data.
4. Demonstrate company-level financial news analysis.
5. Build an interactive financial analytics dashboard.
6. Explore how market sentiment can be monitored programmatically.

---

## 🔮 Future Improvements

Potential improvements include:

* [ ] Add **neutral** sentiment classification
* [ ] Use a dedicated financial NLP model such as **FinBERT**
* [ ] Add historical sentiment tracking
* [ ] Integrate stock price data
* [ ] Compare sentiment with actual price movements
* [ ] Add sentiment scores/probabilities
* [ ] Expand company-to-ticker mapping
* [ ] Add sector-level sentiment analysis
* [ ] Store historical news and predictions
* [ ] Add model evaluation metrics such as accuracy, precision, recall, and F1-score
* [ ] Deploy the dashboard publicly
* [ ] Add automated model retraining
* [ ] Build a time-series analysis of market sentiment

---

## ⚠️ Limitations

This project has several limitations:

* The sentiment model is trained on the provided dataset and may not generalize to every type of financial news.
* Company-to-ticker mapping currently supports only a limited set of companies.
* Headlines without recognized organizations are classified as general market sentiment.
* Sentiment does not directly imply future stock-price movement.
* The project should not be used as an automated trading system or as financial advice.

---

## 📌 Example Use Cases

Market-Mind can be used as a foundation for:

* Financial NLP experiments
* Market sentiment research
* News analytics
* NLP portfolio projects
* Machine-learning demonstrations
* Financial data dashboards
* Future quantitative-finance projects

---

## 🤝 Contributing

Contributions and improvements are welcome.

You can:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit your changes
5. Open a pull request

Ideas for improving the model, dashboard, data pipeline, and financial analytics are especially welcome.

---

## 📄 License

This project does not currently specify a license.

If you intend to make the repository open-source for reuse or contribution, consider adding an appropriate license.

---

## 👨‍💻 Author

**George CA**

GitHub: [@amazephoenix-bit](https://github.com/amazephoenix-bit)

---

## ⭐ Acknowledgements

Built using the Python data-science and machine-learning ecosystem, including **Streamlit, Pandas, Scikit-learn, NLTK, spaCy, and Feedparser**.

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

### 🔗 Repository

[Financial_Sentiment_Market_Mind](https://github.com/amazephoenix-bit/Financial_Sentiment_Market_Mind)

