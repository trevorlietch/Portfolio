import streamlit as st
import pandas as pd

from src.ingest.fetch_prices import fetch_all_prices
from src.analytics.queries import get_portfolio_snapshot

st.set_page_config(page_title="Investment Portfolio", layout="wide")

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

st.title("Investment Portfolio Dashboard")


def format_money(x):
    if pd.isna(x):
        return "-"
    return f"${x:,.2f}"


def render_asset_section(title: str, asset_class: str):
    st.subheader(title)

    data = get_portfolio_snapshot(asset_class)

    if not data:
        st.info(f"No {asset_class} investments found.")
        return

    df = pd.DataFrame(data)

    for col in ["amount", "cost_basis", "current_price", "gain_loss_per_unit", "net_gain_loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate total value
    df["total"] = df["amount"] * df["current_price"]
    # Calculate total net gain/loss
    total_net = df["net_gain_loss"].sum(skipna=True)

    # Top metric
    metric_label = f"Total Net {'Gain' if total_net >= 0 else 'Loss'}"
    st.metric(metric_label, format_money(total_net))

    display_df = df[[
        "symbol",
        "name",
        "amount",
        "total",
        "current_price",
        "net_gain_loss",
    ]].copy()

    # Format for display
    display_df["amount"] = display_df["amount"].map(lambda x: f"{x:,.6f}".rstrip("0").rstrip("."))
    display_df["current_price"] = display_df["current_price"].map(format_money)
    display_df["total"] = display_df["total"].map(format_money)
    display_df["net_gain_loss"] = display_df["net_gain_loss"].map(format_money)

    display_df = display_df.rename(columns={
        "symbol": "Symbol",
        "name": "Name",
        "amount": "Amount",
        "total": "Total",
        "current_price": "Current Price",
        "net_gain_loss": "Net Gain/Loss",
    })

    def color_gain_loss(val):
        if val == "-" or val is None:
            return ""
        try:
            num = float(str(val).replace("$", "").replace(",", ""))
            if num > 0:
                return "color: #16a34a;"   # green
            elif num < 0:
                return "color: #dc2626;"   # red
        except:
            pass
        return ""

    styled_df = display_df.style.map(color_gain_loss, subset=["Net Gain/Loss"])

    st.dataframe(styled_df, width="stretch")


# Refresh prices button
if st.button("Refresh Market Prices"):
    try:
        fetch_all_prices()
        st.success("Market prices updated.")
        st.rerun()
    except Exception as e:
        st.error(f"Could not refresh prices right now: {e}")

st.divider()

render_asset_section("Crypto", "crypto")

st.divider()

render_asset_section("Stocks", "stock")