"""
Monitoring page — Pipeline health, metrics, and cost tracking.

Simple monitoring dashboard showing Lambda metrics,
queue depths, and cost estimation for the deployed pipeline.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.aws_client import get_stats
from utils.visualization import (
    build_gauge_chart,
    build_donut_chart,
    build_line_chart,
    build_bar_chart,
    render_cost_breakdown,
    COLORS,
    SPEECH_TYPE_COLORS,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Monitoring", page_icon="📈", layout="wide")

st.title("📈 Pipeline Monitoring")
st.markdown("Monitor AWS Lambda pipeline health, metrics, and costs.")
st.divider()

# ---------------------------------------------------------------------------
# Fetch stats
# ---------------------------------------------------------------------------

stats = get_stats()

if "error" in stats:
    st.error(f"Unable to fetch pipeline stats: {stats.get('detail', 'Unknown error')}")
    st.info("Make sure the backend is running or check your connection settings.")
    st.stop()

# ---------------------------------------------------------------------------
# Health Overview
# ---------------------------------------------------------------------------

st.markdown("### 💓 Pipeline Health")

health_col1, health_col2, health_col3, health_col4 = st.columns(4)

with health_col1:
    success_rate = stats.get("success_rate", 0)
    st.metric("Success Rate", f"{success_rate:.1f}%",
              delta="Good" if success_rate > 90 else "Needs attention")

with health_col2:
    active = stats.get("active_jobs", 0)
    queue = stats.get("queue_depth", 0)
    st.metric("Active / Queued", f"{active} / {queue}")

with health_col3:
    avg_time = stats.get("average_processing_time", 0)
    st.metric("Avg. Processing", f"{avg_time:.1f}s",
              delta="Fast" if avg_time < 5 else "Slow")

with health_col4:
    dlq = stats.get("dlq_count", 0)
    st.metric("DLQ Messages", dlq,
              delta="Healthy" if dlq == 0 else f"{dlq} need review",
              delta_color="normal" if dlq == 0 else "inverse")

st.divider()

# ---------------------------------------------------------------------------
# Success Rate Gauge
# ---------------------------------------------------------------------------

st.markdown("### 📊 Success Rate")

gauge_col1, gauge_col2 = st.columns([1, 1])

with gauge_col1:
    gauge = build_gauge_chart(
        value=success_rate,
        title="Generation Success Rate",
        max_value=100,
        threshold_good=90,
        threshold_warn=70,
    )
    st.plotly_chart(gauge, use_container_width=True)

with gauge_col2:
    total = stats.get("total_generations", 0)
    successful = stats.get("successful_generations", 0)
    failed = stats.get("failed_generations", 0)

    fig = build_donut_chart(
        labels=["Successful", "Failed"],
        values=[successful, failed],
        title="Success vs Failure",
        colors=[COLORS["success"], COLORS["danger"]],
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Generation Trends
# ---------------------------------------------------------------------------

st.markdown("### 📈 Generation Trends")

daily_data = stats.get("daily_generations", [])
if daily_data:
    dates = [d.get("date", "") for d in daily_data]
    counts = [d.get("count", 0) for d in daily_data]
    trend_fig = build_line_chart(
        x=dates,
        y=counts,
        title="Daily Generations",
        x_label="Date",
        y_label="Count",
        fill=True,
    )
    st.plotly_chart(trend_fig, use_container_width=True)

hourly_data = stats.get("hourly_generations", [])
if hourly_data:
    hours = [f"{h.get('hour', 0):02d}:00" for h in hourly_data]
    h_counts = [h.get("count", 0) for h in hourly_data]
    hourly_fig = build_bar_chart(
        x=hours,
        y=h_counts,
        title="Hourly Distribution",
        x_label="Hour",
        y_label="Generations",
        color=COLORS["info"],
    )
    st.plotly_chart(hourly_fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Generation Sources
# ---------------------------------------------------------------------------

st.markdown("### 🧠 Generation Sources")

source_data = stats.get("generations_by_source", {})
if source_data:
    src_col1, src_col2 = st.columns([2, 1])

    with src_col1:
        source_fig = build_donut_chart(
            labels=[k.title() for k in source_data.keys()],
            values=list(source_data.values()),
            title="Generations by Model Source",
            colors=[COLORS["gemini"], COLORS["gpt2"], COLORS["fallback"]],
        )
        st.plotly_chart(source_fig, use_container_width=True)

    with src_col2:
        total_gen = stats.get("total_generations", 1)
        for source, count in source_data.items():
            pct = count / total_gen * 100 if total_gen > 0 else 0
            icon = {"gemini": "💎", "gpt2": "🧠", "fallback": "📋"}.get(source, "⚡")
            st.markdown(f"**{icon} {source.title()}**: {count:,} ({pct:.1f}%)")

st.divider()

# ---------------------------------------------------------------------------
# Speech Type Distribution
# ---------------------------------------------------------------------------

st.markdown("### 🎭 Speech Type Distribution")

type_data = stats.get("generations_by_type", {})
if type_data:
    type_fig = build_bar_chart(
        x=list(type_data.keys()),
        y=list(type_data.values()),
        title="Generations by Speech Type",
        x_label="Speech Type",
        y_label="Count",
        color=COLORS["primary"],
    )
    st.plotly_chart(type_fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Cost Estimation
# ---------------------------------------------------------------------------

st.markdown("### 💰 Cost Estimation")

cost_data = stats.get("cost_estimate", {})
if cost_data:
    cost_col1, cost_col2 = st.columns([1, 1])

    with cost_col1:
        cost_fig = render_cost_breakdown(cost_data)
        st.plotly_chart(cost_fig, use_container_width=True)

    with cost_col2:
        total_cost = cost_data.get("total", 0)
        st.metric("Estimated Total Cost", f"${total_cost:.2f}", help="Monthly estimated cost")
        st.markdown("#### Cost Breakdown")
        for service, cost in cost_data.items():
            if service != "total":
                st.markdown(f"- **{service.replace('_', ' ').title()}**: ${cost:.2f}")

st.divider()

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Refresh Metrics", use_container_width=True):
        st.rerun()
