# 📈 Stock Price Analysis & Trend Prediction

## Overview

This project analyzes historical stock market data, computes technical indicators, visualizes market trends, and applies machine learning and deep learning models to predict stock price movements.

The system uses real-world stock data from Yahoo Finance and provides insights through data analysis, technical indicators, predictive modeling, and interactive visualizations.

---

## Features

### Data Collection

* Fetches 5 years of historical stock data using Yahoo Finance API
* Supports multiple stocks:

  * AAPL
  * GOOGL
  * MSFT
  * TSLA
  * AMZN

### Data Processing

* Missing value handling
* Daily returns calculation
* Log returns calculation
* Volatility estimation
* Price normalization

### Technical Indicators

* Simple Moving Average (SMA20, SMA50)
* Exponential Moving Average (EMA20)
* Bollinger Bands
* Relative Strength Index (RSI)
* Moving Average Convergence Divergence (MACD)

### Visualizations

* Price comparison charts
* Correlation heatmap
* Moving averages visualization
* Bollinger Bands visualization
* RSI analysis chart
* MACD chart
* Interactive candlestick chart

### Machine Learning

#### Random Forest Classifier

Predicts whether the stock price will increase or decrease on the next trading day.

Features:

* SMA20
* SMA50
* RSI
* MACD
* Volume
* Daily Return
* Volatility

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Deep Learning

#### LSTM Forecasting Model

Predicts future stock closing prices using historical sequences.

Model Architecture:

* Two LSTM Layers
* Dropout Regularization
* Dense Output Layer

Evaluation Metrics:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

---

## Project Structure

```text
Stock-Price-Analysis-And-Trend-Prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── indicators/
│
├── images/
│
├── models/
│
├── src/
│   ├── fetch_data.py
│   ├── clean_data.py
│   ├── technical_indicators.py
│   ├── visualization.py
│   ├── random_forest.py
│   └── lstm_model.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/DHANUSHRI-MUTHUKUMARAN/Stock-Price-Analysis-Trend.git
cd Stock-Price-Analysis-Trend
```

Create environment:

```bash
conda create -n stockml python=3.11
conda activate stockml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1 – Download Stock Data

```bash
python src/fetch_data.py
```

### Step 2 – Clean Data

```bash
python src/clean_data.py
```

### Step 3 – Generate Technical Indicators

```bash
python src/technical_indicators.py
```

### Step 4 – Create Visualizations

```bash
python src/visualization.py
```

### Step 5 – Train Random Forest Model

```bash
python src/random_forest.py
```

### Step 6 – Train LSTM Model

```bash
python src/lstm_model.py
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Scikit-learn
* TensorFlow / Keras
* yfinance

---

## Results

### Random Forest

* Stock Direction Prediction
* Feature Importance Analysis
* Confusion Matrix

### LSTM

* MAE: 6.04
* RMSE: 8.07

---

## Future Improvements

* Streamlit Dashboard
* Real-Time Stock Monitoring
* Multi-Stock Forecasting
* Hyperparameter Optimization
* XGBoost Model Comparison

---

## Author

**Dhanushri Muthukumaran**

