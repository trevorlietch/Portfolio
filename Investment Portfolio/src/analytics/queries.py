from src.db.connection import connect

# For holding calculations, want to sum up all buys and sells 
# for each asset to get net quantity and total invested.
def get_portfolio_snapshot(asset_class: str):
    conn = connect()
    try:
        rows = conn.execute(
            """
            WITH holdings AS (
                SELECT
                    a.id AS asset_id,
                    a.symbol,
                    a.name,
                    a.asset_class,

                    SUM(
                        CASE
                            WHEN t.transaction_type = 'BUY' THEN t.quantity
                            WHEN t.transaction_type = 'SELL' THEN -t.quantity
                            ELSE 0
                        END
                    ) AS amount,

                    SUM(
                        CASE
                            WHEN t.transaction_type = 'BUY' THEN t.price
                            WHEN t.transaction_type = 'SELL' THEN -t.price
                            ELSE 0
                        END
                    ) AS total_invested
                FROM transactions t
                JOIN assets a ON a.id = t.asset_id
                WHERE a.asset_class = ?
                GROUP BY a.id, a.symbol, a.name, a.asset_class
            ),
            latest_prices AS (
                SELECT p.asset_id, p.price AS current_price
                FROM prices p
                JOIN (
                    SELECT asset_id, MAX(price_time) AS max_time
                    FROM prices
                    GROUP BY asset_id
                ) latest
                ON p.asset_id = latest.asset_id
                AND p.price_time = latest.max_time
            )
            SELECT
                h.symbol,
                h.name,
                h.amount,
                h.total_invested,

                CASE
                    WHEN h.amount != 0 THEN h.total_invested / h.amount
                    ELSE 0
                END AS cost_basis,

                lp.current_price,

                CASE
                    WHEN lp.current_price IS NOT NULL AND h.amount != 0
                    THEN lp.current_price - (h.total_invested / h.amount)
                    ELSE NULL
                END AS gain_loss_per_unit,

                CASE
                    WHEN lp.current_price IS NOT NULL
                    THEN (lp.current_price * h.amount) - h.total_invested
                    ELSE NULL
                END AS net_gain_loss

            FROM holdings h
            LEFT JOIN latest_prices lp
                ON h.asset_id = lp.asset_id
            WHERE h.amount != 0
            ORDER BY h.symbol
            """,
            (asset_class,),
        ).fetchall()

        return [dict(r) for r in rows]
    finally:
        conn.close()