from src.db.connection import connect

# Functions to interact with the transactions table in the database
# Requires the necessary information when adding a transaction into the database
def create_transaction(
    account_id: int,
    asset_id: int,
    transaction_type: str,
    quantity: float,
    price: float,
    transaction_time: str,
):
    conn = connect()
    try:
        conn.execute(
            """
            INSERT INTO transactions (
                account_id, asset_id, transaction_type,
                quantity, price, transaction_date
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (account_id, asset_id, transaction_type, quantity, price, transaction_time),
        )
        conn.commit()
    finally:
        conn.close()


def list_transactions(limit: int = 200):
    conn = connect()
    try:
        rows = conn.execute(
            """
            SELECT
                t.id,
                t.transaction_date AS transaction_date,
                a.name AS account_name,
                s.symbol AS symbol,
                s.asset_class AS asset_class,
                t.transaction_type,
                t.quantity AS quantity,
                t.price,
                t.created_at
            FROM transactions t
            JOIN accounts a ON a.id = t.account_id
            JOIN assets s ON s.id = t.asset_id
            ORDER BY t.transaction_date DESC, t.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_transaction(txn_id: int):
    conn = connect()
    try:
        conn.execute("DELETE FROM transactions WHERE id = ?", (txn_id,))
        conn.commit()
    finally:
        conn.close()