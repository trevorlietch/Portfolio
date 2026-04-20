import os
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = "data/portfolio.db"

def get_db_path():
    return Path(os.getenv("PORTFOLIO_DB_PATH", str(DEFAULT_DB_PATH)))

# use "from src.db.connection import connect" in other files to connect to the database
# every function that interacts will use:
# conn = connect()
# try: conn.execute() and conn.commit()
# finally: conn.close()
def connect():
    db_path = get_db_path()
    # Prevents system errors if the directory doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON;")
    connection.execute("PRAGMA journal_mode = WAL;")
    connection.execute("PRAGMA synchronous = NORMAL;")

    return connection