import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib

# Load data
df = pd.read_csv("data/stock_data.csv")

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date")

# Create features
df["Previous_Close"] = df["Close"].shift(1)
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()
df["Daily_Return"] = df["Close"].pct_change()

# Target = next day's closing price
df["Target"] = df["Close"].shift(-1)

# Remove missing values
df = df.dropna()

# Features
features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Previous_Close",
    "MA_5",
    "MA_10",
    "Daily_Return"
]

X = df[features]
y = df["Target"]

# Time-based split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# XGBoost model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Model Training Completed!")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, "models/stock_model.pkl")

print("Model saved successfully!")