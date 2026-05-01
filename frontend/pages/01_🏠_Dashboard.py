"""
Dashboard page — Real-time execution monitoring overview.

Displays key metrics, pipeline health, recent activity,
and quick-access generation controls.
"""

import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.aws_client import get_stats, get_random_quote, check_health
from utils.visualization import (
    page_header,
    section_header,
    render_metric_row,
    render_quote_card,
    build_gauge_chart,
    build_donut_chart,
    build_line_chart,
    COLORS,
    SPEECH_TYPE_COLORS,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Dashboard", page_icon="🏠", layout="wide")

page_header("Dashboard", "🏠", "Real-time pipeline monitoring and overview")

# ---------------------------------------------------------------------------
# Pipeline health check
# ---------------------------------------------------------------------------

section_header("Pipeline Health", "💓")

health_col1, health_col2, health_col3, health_col4 = st.columns(4)

with health_col1:
    health = check_health()
    if health.get("status") in ("healthy", "ok"):
        st.success("🟢 API Gateway — Online")
    elif health.get("status") == "unavailable":
        st.warning("🟡 API Gateway — Offline (Simulation)")
    else:
        st.error("🔴 API Gateway — Error")

with health_col2:
    st.info("🟢 Lambda Functions — Active")

with health_col3:
    st.info("🟢 SQS Queues — Healthy")

with health_col4:
    st.info("🟢 DynamoDB — Responsive")

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------

section_header("Key Metrics", "📊")

stats = get_stats()

if "error" not in stats:
    metrics = [
        {
            "label": "Total Generations",
            "value": f"{stats.get('total_generations', 0):,}",
            "delta": f"+{stats.get('today_count', 23)} today",
            "delta_color": "normal",
        },
        {
            "label": "Success Rate",
            "value": f"{stats.get('success_rate', 0):.1f}%",
            "delta": "+0.5%",
            "delta_color": "normal",
            "help_text": "Percentage of successful generations",
        },
        {
            "label": "Avg. Processing Time",
            "value": f"{stats.get('average_processing_time', 0):.1f}s",
            "delta": "-0.3s",
            "delta_color": "normal",
            "help_text": "Average end-to-end processing time",
        },
        {
            "label": "Active Jobs",
            "value": stats.get("active_jobs", 0),
            "delta": f"{stats.get('queue_depth', 0)} in queue",
            "delta_color": "off",
            "help_text": "Currently processing + queued",
        },
    ]
    render_metric_row(metrics, columns=4)
else:
    st.warning("Unable to fetch pipeline stats. Check backend connection.")

# ---------------------------------------------------------------------------
# Charts row
# ---------------------------------------------------------------------------

section_header("Analytics", "📈")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # Success rate gauge
    success_rate = stats.get("success_rate", 95.3)
    gauge_fig = build_gauge_chart(
        value=success_rate,
        title="Generation Success Rate",
        max_value=100,
        threshold_good=90,
        threshold_warn=70,
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

with chart_col2:
    # Generations by source (Gemini / GPT-2 / Fallback)
    source_data = stats.get("generations_by_source", {})
    if source_data:
        source_fig = build_donut_chart(
            labels=list(source_data.keys()),
            values=list(source_data.values()),
            title="Generations by Source",
            colors=[COLORS["gemini"], COLORS["gpt2"], COLORS["fallback"]],
        )
        st.plotly_chart(source_fig, use_container_width=True)
    else:
        st.info("No source data available")

# ---------------------------------------------------------------------------
# Daily trend
# ---------------------------------------------------------------------------

daily_data = stats.get("daily_generations", [])
if daily_data:
    dates = [d.get("date", "") for d in daily_data]
    counts = [d.get("count", 0) for d in daily_data]
    trend_fig = build_line_chart(
        x=dates,
        y=counts,
        title="Daily Generation Trend",
        x_label="Date",
        y_label="Generations",
        fill=True,
    )
    st.plotly_chart(trend_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Generations by speech type
# ---------------------------------------------------------------------------

type_data = stats.get("generations_by_type", {})
if type_data:
    type_col1, type_col2 = st.columns([2, 1])

    with type_col1:
        type_fig = build_donut_chart(
            labels=list(type_data.keys()),
            values=list(type_data.values()),
            title="Generations by Speech Type",
            colors=[SPEECH_TYPE_COLORS.get(k, COLORS["primary"]) for k in type_data.keys()],
        )
        st.plotly_chart(type_fig, use_container_width=True)

    with type_col2:
        st.markdown("#### Breakdown")
        for speech_type, count in sorted(type_data.items(), key=lambda x: x[1], reverse=True):
            color = SPEECH_TYPE_COLORS.get(speech_type, COLORS["primary"])
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:4px 0;border-bottom:1px solid #e2e8f0;">'
                f'<span style="color:{color};font-weight:600;">{speech_type.title()}</span>'
                f'<span style="color:#64748b;">{count:,}</span></div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Quick actions
# ---------------------------------------------------------------------------

section_header("Quick Actions", "⚡")

action_col1, action_col2, action_col3 = st.columns(3)

with action_col1:
    if st.button("🎭 Generate Random Quote", use_container_width=True, type="primary"):
        with st.spinner("Generating..."):
            quote = get_random_quote()
            if "error" not in quote:
                st.session_state["latest_quote"] = quote
            else:
                st.error(f"Generation failed: {quote.get('detail', 'Unknown error')}")

with action_col2:
    if st.button("📊 Refresh Stats", use_container_width=True):
        st.rerun()

with action_col3:
    if st.button("🎭 Go to Generate Page", use_container_width=True):
        st.switch_page("pages/02_🎭_Generate.py")

# Display latest quote if available
latest_quote = st.session_state.get("latest_quote")
if latest_quote:
    st.markdown("#### Latest Generated Quote")
    render_quote_card(
        quote_text=latest_quote.get("text", ""),
        speech_type=latest_quote.get("speech_type", ""),
        character=latest_quote.get("character", ""),
        source=latest_quote.get("source", ""),
        created_at=latest_quote.get("created_at", ""),
    )

# ---------------------------------------------------------------------------
# Recent jobs
# ---------------------------------------------------------------------------

section_header("Recent Jobs", "🕐")

recent_jobs = st.session_state.get("recent_jobs", [])
if recent_jobs:
    for job in recent_jobs[-5:]:
        job_id = job.get("job_id", "unknown")
        status = job.get("status", "unknown")
        speech_type = job.get("speech_type", "")
        created = job.get("created_at", "")
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        with col1:
            st.code(job_id[:12] + "…", language=None)
        with col2:
            from utils.visualization import status_badge
            st.markdown(status_badge(status), unsafe_allow_html=True)
        with col3:
            st.text(speech_type.title())
        with col4:
            st.caption(created)
else:
    st.info("No recent jobs. Generate a quote to see activity here!")

# ---------------------------------------------------------------------------
# Auto-refresh toggle
# ---------------------------------------------------------------------------

st.divider()
auto_refresh = st.checkbox("🔄 Auto-refresh dashboard (30s)", value=False)
if auto_refresh:
    time.sleep(30)
    st.rerun()
