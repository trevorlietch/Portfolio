import streamlit as st
import pandas as pd
from datetime import date

from src.db.connection import connect
from src.db.tables.transactions import create_transaction, list_transactions, delete_transaction

st.title("Transactions")

# Load account and assets 
def load_accounts():
    conn = connect()
    try:
        rows = conn.execute("SELECT id, name, account_type FROM accounts ORDER BY name").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def load_assets():
    conn = connect()
    try:
        rows = conn.execute("SELECT id, symbol, name, asset_class FROM assets ORDER BY symbol").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

accounts = load_accounts()
assets = load_assets()

if not accounts:
    st.warning("No accounts found. Add an account first on the Accounts page.")
    st.stop()

if not assets:
    st.warning("No assets found. Add an asset first on the Assets page.")
    st.stop()

# Build selectbox options
account_labels = {a["id"]: f'{a["name"]}' for a in accounts}
asset_labels = {s["id"]: f'{s["symbol"]} - {s["name"]}' for s in assets}

st.subheader("Add a transaction")

with st.form("add_txn"):
    account_id = st.selectbox(
        "Account",
        options=list(account_labels.keys()),
        format_func=lambda i: account_labels[i],
    )

    asset_id = st.selectbox(
        "Asset",
        options=list(asset_labels.keys()),
        format_func=lambda i: asset_labels[i],
    )

    transaction_type = st.selectbox("Type", ["BUY", "SELL"])

    col1, col2 = st.columns(2)
    with col1:
        quantity = st.number_input("Quantity", min_value=0.0, value=1.0, step=0.0001, format="%.6f")
    with col2:
        price = st.number_input("Total amount paid (USD)", min_value=0.0, value=0.0, step=0.01, format="%.2f")

    txn_date = st.date_input("Transaction date", value=date.today())

    # save as YYYY-MM-DD
    transaction_date = txn_date.strftime("%Y-%m-%d")

    submitted = st.form_submit_button("Add transaction")

if submitted:
    # Basic validation
    if quantity <= 0:
        st.error("Quantity must be > 0.")
    elif price <= 0:
        st.error("Price must be > 0.")
    else:
        create_transaction(
            account_id=int(account_id),
            asset_id=int(asset_id),
            transaction_type=transaction_type,
            quantity=float(quantity),
            price=float(price),
            transaction_time=transaction_date,
        )
        st.success("Transaction added.")
        st.rerun()

st.divider()

st.subheader("Recent transactions")

txns = list_transactions(limit=200)
if not txns:
    st.info("No transactions yet.")
else:
    df = pd.DataFrame(txns)

    preferred = ["id", "symbol","transaction_date", "account_name", "transaction_type", "quantity", "price"]
    df = df[[c for c in preferred if c in df.columns]]
    st.dataframe(df, use_container_width=True)

    # Delete UI
    st.caption("Delete a transaction")
    txn_ids = [t["id"] for t in txns]
    to_delete = st.selectbox("Transaction ID to delete", txn_ids)
    if st.button("Delete selected transaction"):
        delete_transaction(int(to_delete))
        st.success("Deleted.")
        st.rerun()