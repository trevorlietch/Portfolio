import streamlit as st
from src.db.tables.assets import create_asset, list_assets

st.set_page_config(page_title="Assets", layout="wide")

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

st.title("Assets")

# Inserts new asset into the database
# This is the type of investment like "BTC" or "AAPL"
# The transaction page will pull from this and allow you to input the amount and price
with st.form("add_asset"):
    symbol = st.text_input("Symbol", placeholder="BTC or AAPL")
    name = st.text_input("Name", placeholder="Bitcoin or Apple")
    asset_class = st.selectbox("Asset class", ["crypto", "stock", "other"])
    submitted = st.form_submit_button("Add asset")

if submitted:
    if not symbol or not name:
        st.error("Please fill all fields.")
    else:
        create_asset(symbol, name, asset_class)
        st.success("Asset added.")
        st.rerun()

st.divider()

st.subheader("Your assets")
assets = list_assets()

if not assets:
    st.info("No assets yet. Add one above.")
else:
    for a in assets:
        col1, col2 = st.columns([2, 1])
        col1.write(f"**{a['symbol']}** — {a['name']}")
        col2.caption(a['asset_class'])