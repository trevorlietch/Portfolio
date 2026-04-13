from datetime import datetime

from src.db.connection import connect
from src.ingest.prices.coingecko import get_crypto_prices
from src.ingest.prices.yahoo import get_stock_price

def fetch_all_prices():
    conn = connect()
    try:
        assets = conn.execute("""
            SELECT id, symbol, asset_class
            FROM assets
        """).fetchall()

        crypto_assets = [a for a in assets if a["asset_class"].lower() == "crypto"]
        stock_assets = [a for a in assets if a["asset_class"].lower() == "stock"]

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Fetch all crypto prices in one request
        crypto_symbols = [a["symbol"] for a in crypto_assets]
        crypto_prices = get_crypto_prices(crypto_symbols)

        for asset in crypto_assets:
            symbol = asset["symbol"].upper()
            price = crypto_prices.get(symbol)
            if price is None:
                continue

            conn.execute("""
                INSERT INTO prices (asset_id, price, price_time, source)
                VALUES (?, ?, ?, ?)
            """, (asset["id"], price, now, "coingecko"))

        # Fetch stocks one at a time
        for asset in stock_assets:
            price = get_stock_price(asset["symbol"])
            if price is None:
                continue

            conn.execute("""
                INSERT INTO prices (asset_id, price, price_time, source)
                VALUES (?, ?, ?, ?)
            """, (asset["id"], price, now, "yahoo"))

        conn.commit()
    finally:
        conn.close()