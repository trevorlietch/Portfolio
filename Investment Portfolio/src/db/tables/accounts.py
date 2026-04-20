# Takes information from /app/pages/accounts.py and inserts that information into the database

from src.db.connection import connect

# Inserts a new account into the database when you click the "Add account" button
def create_account(name: str, account_type: str) -> None:
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO accounts (name, account_type) VALUES (?, ?)",
            (name.strip(), account_type.strip())
        )
        conn.commit()
    finally:
        conn.close()

# Shows the list below the form on the accounts page
def list_accounts():
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id, name, account_type, created_at FROM accounts ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

# Deletes an account when you click the delete button
def delete_account(account_id: int) -> None:
    conn = connect()
    try:
        conn.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
        conn.commit()
    finally:
        conn.close()