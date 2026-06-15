import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Create output folder
os.makedirs("data/processed", exist_ok=True)

tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

for ticker in tickers:

    print(f"Processing {ticker}...")

    # Read raw data
    df = pd.read_csv(f"data/raw/{ticker}.csv")

    # Remove extra header rows created by yfinance
    df = df.iloc[2:].reset_index(drop=True)

    # Rename columns
    df.columns = [
        "Date",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]

    # Convert numeric columns
    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Daily Return
    df["Daily_Return"] = df["Close"].pct_change()

    # Log Return
    df["Log_Return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    # Volatility (30-day rolling std)
    df["Volatility"] = (
        df["Daily_Return"]
        .rolling(window=30)
        .std()
    )

    # Normalize Close Price
    scaler = MinMaxScaler()

    df["Close_Normalized"] = scaler.fit_transform(
        df[["Close"]]
    )

    # Save cleaned data
    output_path = f"data/processed/{ticker}.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")

print("\nAll stocks processed successfully!")