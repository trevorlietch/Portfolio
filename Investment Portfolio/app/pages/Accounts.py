import streamlit as st
from src.db.tables.accounts import create_account, list_accounts, delete_account

st.title("Accounts")

with st.form("add_account"):
    name = st.text_input("Account name", placeholder="Schwab, Coinbase, etc.")
    account_type = st.selectbox("Account type", ["stocks", "crypto", "other"])
    submitted = st.form_submit_button("Add account")

if submitted:
    if not name.strip():
        st.error("Please enter an account name.")
    else:
        create_account(name, account_type)
        st.success("Account added.")
        st.rerun()

st.divider()

st.subheader("Your accounts")
accounts = list_accounts()

if not accounts:
    st.info("No accounts yet. Add one above.")
else:
    for a in accounts:
        col1, col2, col3 = st.columns([3, 2, 2])
        col1.write(f"**{a['name']}**")
        col2.write(a["account_type"])

        if col3.button("Delete", key=f"del_{a['id']}"):
            delete_account(a["id"])
            st.rerun()