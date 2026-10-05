import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Stock Analysis & Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Analysis & Prediction")
st.write("Educational stock-price analysis using historical data and machine learning.")

st.warning(
    "For learning and educational purposes only. "
    "This application is not trading or investment advice."
)

# Load data
df = pd.read_csv("data/stock_data.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Load model
model = joblib.load("models/stock_model.pkl")

# Create features
df["Previous_Close"] = df["Close"].shift(1)
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()
df["Daily_Return"] = df["Close"].pct_change()

df = df.dropna()

# Sidebar
st.sidebar.header("Stock Information")

start_date = st.sidebar.date_input(
    "Start Date",
    df["Date"].min().date()
)

end_date = st.sidebar.date_input(
    "End Date",
    df["Date"].max().date()
)

filtered_df = df[
    (df["Date"].dt.date >= start_date) &
    (df["Date"].dt.date <= end_date)
]

# Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Latest Close",
        f"{df['Close'].iloc[-1]:.2f}"
    )

with col2:
    st.metric(
        "Highest Close",
        f"{filtered_df['Close'].max():.2f}"
    )

with col3:
    st.metric(
        "Lowest Close",
        f"{filtered_df['Close'].min():.2f}"
    )

# Price chart
st.subheader("Historical Stock Price")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Close"],
        mode="lines",
        name="Closing Price"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# Moving averages
st.subheader("Moving Average")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["Close"],
        mode="lines",
        name="Close"
    )
)

fig2.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["MA_5"],
        mode="lines",
        name="5-Day MA"
    )
)

fig2.add_trace(
    go.Scatter(
        x=filtered_df["Date"],
        y=filtered_df["MA_10"],
        mode="lines",
        name="10-Day MA"
    )
)

fig2.update_layout(
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig2, use_container_width=True)

# Prediction
st.subheader("Next-Day Price Prediction")

latest = df.iloc[-1]

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

input_data = pd.DataFrame(
    [[latest[f] for f in features]],
    columns=features
)

prediction = model.predict(input_data)[0]

st.success(
    f"Estimated next closing price: {prediction:.2f}"
)

# Raw data
st.subheader("Historical Data")

st.dataframe(
    filtered_df[
        ["Date", "Open", "High", "Low", "Close", "Volume"]
    ],
    use_container_width=True
)