import streamlit as st
from src.db.tables.accounts import create_account, list_accounts, delete_account

st.set_page_config(page_title="Accounts", layout="wide")

# Sidebar formatting
st.markdown("""
<style>
    /* Space out sidebar navigation items */
    div[data-testid="stSidebarNav"] li {
        margin-bottom: 12px !important;
        padding: 6px 0 !important;
    }
    div[data-testid="stSidebarNav"] a {
        padding: 10px 16px !important;
        border-radius: 8px !important;
        transition: background-color 0.2s !important;
        text-align: center !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    div[data-testid="stSidebarNav"] a:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Style buttons */
    button {
        font-size: 18px !important;
        padding: 12px 24px !important;
        text-align: center !important;
        width: 100% !important;
    }
    
    button p {
        text-align: center !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

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
            try:
                delete_account(a["id"])
                st.success("Account deleted.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not delete account: {e}")