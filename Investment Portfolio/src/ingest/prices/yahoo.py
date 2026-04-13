# Importing Stock Prices from Yahoo Finance API
import yfinance as yf

def get_stock_price(symbol: str):
    """
    Fetch latest stock price using Yahoo Finance
    """

    ticker = yf.Ticker(symbol)

    data = ticker.history(period="1d")

    if data.empty:
        return None

    return float(data["Close"].iloc[-1])