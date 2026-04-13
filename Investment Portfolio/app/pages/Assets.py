import streamlit as st
from src.db.tables.assets import create_asset, list_assets

st.title("Assets")

# Inserts new asset into the database
# This is the type of investment like "BTC" or "AAPL"
# The transaction page will pull from this and allow you to input the ammount and price
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

for a in assets:
    st.write(f"{a['symbol']} - {a['name']} ({a['asset_class']})")