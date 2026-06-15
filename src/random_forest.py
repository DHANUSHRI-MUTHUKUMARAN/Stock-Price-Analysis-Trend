import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

ticker = "AAPL"

df = pd.read_csv(f"data/indicators/{ticker}.csv")

# Create target
df["Target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)

features = [
    "SMA20",
    "SMA50",
    "RSI",
    "MACD",
    "Volume",
    "Daily_Return",
    "Volatility"
]

# Remove missing rows
df = df.dropna()

X = df[features]
y = df["Target"]

# Time-based split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

rf.fit(X_train, y_train)

# Predictions
predictions = rf.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-" * 30)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Save model
joblib.dump(
    rf,
    "models/rf_model.pkl"
)

print("\nModel saved.")

# Feature Importance
importance = pd.Series(
    rf.feature_importances_,
    index=features
)

importance.sort_values().plot(
    kind="barh",
    figsize=(8,5)
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
    "images/feature_importance.png"
)

plt.close()

print("Feature importance saved.")

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.savefig(
    "images/confusion_matrix.png"
)

plt.close()

print("Confusion matrix saved.")