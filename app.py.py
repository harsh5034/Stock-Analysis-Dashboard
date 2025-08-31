from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
import json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Load the trained LSTM model
MODEL_PATH = "simple_lstm.keras"
if os.path.exists(MODEL_PATH):
    lstm_model = load_model(MODEL_PATH)
else:
    lstm_model = None

scaler = MinMaxScaler(feature_range=(0, 1))


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    ticker = request.form["ticker"].upper()

    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")

        if hist.empty:
            return render_template("error.html", message=f"No data found for {ticker}")

        # Stock + Company Info
        stock_info = {
            "name": stock.info.get("longName", ticker),
            "symbol": ticker,
            "current_price": stock.info.get("currentPrice", "N/A"),
            "market_cap": stock.info.get("marketCap", "N/A"),
            "high_52week": stock.info.get("fiftyTwoWeekHigh", "N/A"),
            "low_52week": stock.info.get("fiftyTwoWeekLow", "N/A"),
            "book_value": stock.info.get("bookValue", "N/A"),
            "pe_ratio": stock.info.get("trailingPE", "N/A"),
            "roe": stock.info.get("returnOnEquity", "N/A"),
            "roce": stock.info.get("returnOnAssets", "N/A"),
            "face_value": stock.info.get("lastDividendValue", "N/A"),

            # Company Details
            "sector": stock.info.get("sector", "N/A"),
            "industry": stock.info.get("industry", "N/A"),
            "country": stock.info.get("country", "N/A"),
            "website": stock.info.get("website", "N/A"),
            "summary": stock.info.get("longBusinessSummary", "N/A"),
        }

        # Plot actual stock prices
        actual_chart = go.Figure()
        actual_chart.add_trace(go.Scatter(
            x=hist.index, y=hist["Close"], mode="lines", name="Closing Price"
        ))
        actual_chart.update_layout(
            title=f"{ticker} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white"
        )
        actual_chart_json = json.dumps(actual_chart, cls=PlotlyJSONEncoder)

        # LSTM Predictions
        predicted_prices = []
        if lstm_model is not None:
            close_data = hist[["Close"]].values
            scaled_data = scaler.fit_transform(close_data)

            if len(scaled_data) >= 60:
                last_60 = scaled_data[-60:]
                X_test = np.array([last_60])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                predicted_scaled = lstm_model.predict(X_test)
                predicted_price = scaler.inverse_transform(predicted_scaled)
                predicted_prices = predicted_price.flatten().tolist()

        return render_template(
            "dashboard.html",
            stock_info=stock_info,
            actual_chart=actual_chart_json,
            predicted_prices=predicted_prices
        )

    except Exception as e:
        return render_template("error.html", message=str(e))


if __name__ == "__main__":
    app.run(debug=True)
