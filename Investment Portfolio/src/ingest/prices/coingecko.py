# Importing Crypto Prices from CoinGecko API
import requests

COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"

SYMBOL_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "ADA": "cardano",
    "SOL": "solana",
    "XRP": "ripple",
    "BONK": "bonk",
    "BRETT": "based-brett",
}

def get_crypto_prices(symbols: list[str]) -> dict[str, float]:
    ids = []
    reverse_map = {}

    for symbol in symbols:
        coin_id = SYMBOL_MAP.get(symbol.upper())
        if coin_id:
            ids.append(coin_id)
            reverse_map[coin_id] = symbol.upper()

    if not ids:
        return {}

    response = requests.get(
        COINGECKO_URL,
        params={
            "ids": ",".join(ids),
            "vs_currencies": "usd",
        },
        timeout=10,
    )

    if response.status_code == 429:
        return {}

    response.raise_for_status()
    data = response.json()

    prices = {}
    for coin_id, payload in data.items():
        symbol = reverse_map.get(coin_id)
        if symbol and "usd" in payload:
            prices[symbol] = payload["usd"]

    return prices