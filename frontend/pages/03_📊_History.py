"""
History page — Browse and search previously generated quotes.

Simple interface to view generation history, filter by type,
and review past outputs from the pipeline.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.aws_client import get_quotes, get_stats
from utils.visualization import render_quote_card, build_donut_chart, SPEECH_TYPE_COLORS

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="History", page_icon="📊", layout="wide")

st.title("📊 Generation History")
st.markdown("Browse and search previously generated anime quotes.")
st.divider()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

with filter_col1:
    speech_filter = st.selectbox(
        "Filter by Speech Type",
        ["All", "motivational", "villain", "philosophical", "heroic", "emotional", "comedic"],
    )

with filter_col2:
    page_num = st.number_input("Page", min_value=1, value=1, step=1)

with filter_col3:
    page_size = st.selectbox("Per Page", [10, 20, 50], index=1)

# ---------------------------------------------------------------------------
# Fetch quotes
# ---------------------------------------------------------------------------

filter_type = None if speech_filter == "All" else speech_filter
quotes_data = get_quotes(page=page_num, page_size=page_size, speech_type=filter_type)

if "error" in quotes_data:
    st.error(f"Failed to load quotes: {quotes_data.get('detail', 'Unknown error')}")
    quotes_list = []
else:
    quotes_list = quotes_data.get("quotes", [])
    total = quotes_data.get("total", 0)
    total_pages = quotes_data.get("total_pages", 1)

    st.caption(f"Showing {len(quotes_list)} of {total} quotes (Page {page_num}/{total_pages})")

# ---------------------------------------------------------------------------
# Display quotes
# ---------------------------------------------------------------------------

if quotes_list:
    for quote in quotes_list:
        render_quote_card(
            quote_text=quote.get("text", quote.get("quote", "")),
            speech_type=quote.get("speech_type", ""),
            character=quote.get("character", ""),
            source=quote.get("source", ""),
            created_at=quote.get("created_at", ""),
        )
else:
    st.info("No quotes found. Generate some quotes first! 🎭")

# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

if quotes_list and total_pages > 1:
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if page_num > 1 and st.button("⬅️ Previous"):
            st.rerun()
    with nav_col3:
        if page_num < total_pages and st.button("Next ➡️"):
            st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Stats summary
# ---------------------------------------------------------------------------

st.markdown("### 📈 Generation Stats")

stats = get_stats()
if "error" not in stats:
    type_data = stats.get("generations_by_type", {})
    if type_data:
        fig = build_donut_chart(
            labels=list(type_data.keys()),
            values=list(type_data.values()),
            title="Generations by Speech Type",
            colors=[SPEECH_TYPE_COLORS.get(k, "#6366f1") for k in type_data.keys()],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Quick stats table
    source_data = stats.get("generations_by_source", {})
    if source_data:
        st.markdown("#### Generation Sources")
        source_df = pd.DataFrame([
            {"Source": k.title(), "Count": v, "Percentage": f"{v/stats.get('total_generations', 1)*100:.1f}%"}
            for k, v in source_data.items()
        ])
        st.dataframe(source_df, use_container_width=True, hide_index=True)
