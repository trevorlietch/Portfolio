import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.db.connection import connect

conn = connect()
rows = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()
conn.close()

print([row[0] for row in rows])

# To run: python .\tests\verify_db.py
# Expected output: ['accounts', 'assets', 'prices', 'transactions']
# This verifies that you create tables with those names
# Also might see 'sqlite_sequence' since we have AUTOINCREMENT