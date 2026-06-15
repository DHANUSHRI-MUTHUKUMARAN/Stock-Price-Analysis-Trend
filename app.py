import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Analysis Dashboard")

st.title("📈 Stock Price Analysis & Prediction")

ticker = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
)

df = pd.read_csv(f"data/indicators/{ticker}.csv")

st.subheader(f"{ticker} Overview")

st.metric(
    "Current Price",
    f"${df['Close'].iloc[-1]:.2f}"
)

# Candlestick
fig = go.Figure(
    data=[
        go.Candlestick(
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )
    ]
)

st.plotly_chart(fig, use_container_width=True)

# RSI
st.subheader("RSI")

st.line_chart(df["RSI"])

# MACD
st.subheader("MACD")

st.line_chart(
    df[["MACD", "Signal"]]
)

# Bollinger Bands
st.subheader("Bollinger Bands")

st.line_chart(
    df[
        [
            "Close",
            "Upper_Band",
            "Lower_Band"
        ]
    ]
)