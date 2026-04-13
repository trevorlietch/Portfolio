from src.db.connection import connect

conn = connect()

# Check if BRETT asset exists
brett = conn.execute('SELECT id, symbol, name FROM assets WHERE symbol = ?', ('BRETT',)).fetchone()
print('BRETT Asset:', dict(brett) if brett else 'NOT FOUND')

# Check BRETT transactions
if brett:
    transactions = conn.execute('SELECT id, quantity, price FROM transactions WHERE asset_id = ?', (brett[0],)).fetchall()
    print('BRETT Transactions:', [dict(t) for t in transactions])
    
    # Check BRETT prices
    prices = conn.execute('SELECT id, price, price_time FROM prices WHERE asset_id = ? ORDER BY price_time DESC LIMIT 3', (brett[0],)).fetchall()
    print('BRETT Prices:', [dict(p) for p in prices])

conn.close()
