from src.db.connection import connect

def create_asset(symbol: str, name: str, asset_class: str):
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO assets (symbol, name, asset_class) VALUES (?, ?, ?)",
            (symbol.upper(), name.strip(), asset_class.strip())
        )
        conn.commit()
    finally:
        conn.close()

def list_assets():
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id, symbol, name, asset_class FROM assets ORDER BY symbol"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()