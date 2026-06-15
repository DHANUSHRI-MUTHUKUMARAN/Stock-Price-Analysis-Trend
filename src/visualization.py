import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os

os.makedirs("images", exist_ok=True)

tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

# ---------------------------
# 1. Price Comparison Chart
# ---------------------------

plt.figure(figsize=(12,6))

for ticker in tickers:
    df = pd.read_csv(f"data/indicators/{ticker}.csv")
    plt.plot(df["Close"], label=ticker)

plt.title("Stock Closing Price Comparison")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()

plt.savefig("images/price_comparison.png")
plt.close()

print("Saved: price_comparison.png")

# ---------------------------
# 2. Correlation Heatmap
# ---------------------------

returns = pd.DataFrame()

for ticker in tickers:
    df = pd.read_csv(f"data/indicators/{ticker}.csv")
    returns[ticker] = df["Daily_Return"]

corr = returns.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True)

plt.title("Stock Return Correlation")

plt.savefig("images/correlation_heatmap.png")
plt.close()

print("Saved: correlation_heatmap.png")

# ---------------------------
# 3. Individual Stock Charts
# ---------------------------

ticker = "AAPL"

df = pd.read_csv(f"data/indicators/{ticker}.csv")

# SMA + EMA

plt.figure(figsize=(12,6))

plt.plot(df["Close"], label="Close")
plt.plot(df["SMA20"], label="SMA20")
plt.plot(df["SMA50"], label="SMA50")
plt.plot(df["EMA20"], label="EMA20")

plt.legend()

plt.title(f"{ticker} Moving Averages")

plt.savefig("images/moving_averages.png")
plt.close()

print("Saved: moving_averages.png")

# ---------------------------
# 4. Bollinger Bands
# ---------------------------

plt.figure(figsize=(12,6))

plt.plot(df["Close"], label="Close")
plt.plot(df["Upper_Band"], label="Upper Band")
plt.plot(df["Lower_Band"], label="Lower Band")

plt.legend()

plt.title(f"{ticker} Bollinger Bands")

plt.savefig("images/bollinger_bands.png")
plt.close()

print("Saved: bollinger_bands.png")

# ---------------------------
# 5. RSI
# ---------------------------

plt.figure(figsize=(12,4))

plt.plot(df["RSI"])

plt.axhline(70, linestyle="--")
plt.axhline(30, linestyle="--")

plt.title(f"{ticker} RSI")

plt.savefig("images/rsi_chart.png")
plt.close()

print("Saved: rsi_chart.png")

# ---------------------------
# 6. MACD
# ---------------------------

plt.figure(figsize=(12,4))

plt.plot(df["MACD"], label="MACD")
plt.plot(df["Signal"], label="Signal")

plt.legend()

plt.title(f"{ticker} MACD")

plt.savefig("images/macd_chart.png")
plt.close()

print("Saved: macd_chart.png")

# ---------------------------
# 7. Candlestick Chart
# ---------------------------

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

fig.write_html("images/candlestick.html")

print("Saved: candlestick.html")