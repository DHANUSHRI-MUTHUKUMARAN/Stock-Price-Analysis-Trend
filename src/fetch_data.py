import yfinance as yf
import os

os.makedirs("data/raw", exist_ok=True)

tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

for ticker in tickers:
    df = yf.download(
        ticker,
        start="2021-01-01",
        end="2026-01-01",
        auto_adjust=True
    )

    df.to_csv(f"data/raw/{ticker}.csv")

    print(f"{ticker} saved")