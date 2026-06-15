import pandas as pd
import numpy as np
import os

tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

os.makedirs("data/indicators", exist_ok=True)

for ticker in tickers:

    print(f"Processing {ticker}...")

    df = pd.read_csv(f"data/processed/{ticker}.csv")

    # SMA
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    # EMA
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # Bollinger Bands
    rolling_std = df["Close"].rolling(window=20).std()

    df["Upper_Band"] = df["SMA20"] + (2 * rolling_std)
    df["Lower_Band"] = df["SMA20"] - (2 * rolling_std)

    # RSI
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Save
    output_file = f"data/indicators/{ticker}.csv"

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")

print("\nTechnical indicators generated successfully!")