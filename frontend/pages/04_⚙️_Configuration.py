"""
Configuration page — Manage AWS resource settings and API configuration.

Simple interface for configuring backend connection, viewing
AWS resource details, and managing environment settings.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.aws_client import get_config, check_health

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Configuration", page_icon="⚙️", layout="wide")

st.title("⚙️ Configuration")
st.markdown("Manage AWS resource settings and API configuration.")
st.divider()

# ---------------------------------------------------------------------------
# Backend Connection
# ---------------------------------------------------------------------------

st.markdown("### 🔌 Backend Connection")

current_url = st.session_state.get("api_base_url", "http://localhost:8000/api/v1")
current_key = st.session_state.get("api_key", "")

with st.form("connection_form"):
    new_url = st.text_input("API Base URL", value=current_url)
    new_key = st.text_input("API Key", value=current_key, type="password")

    if st.form_submit_button("💾 Save Connection Settings", use_container_width=True):
        st.session_state["api_base_url"] = new_url
        st.session_state["api_key"] = new_key
        st.success("✅ Settings saved!")

# Test connection
if st.button("🔍 Test Connection", use_container_width=True):
    health = check_health()
    if health.get("status") in ("healthy", "ok"):
        st.success("🟢 Backend is reachable and healthy!")
    elif health.get("status") == "unavailable":
        st.warning("🟡 Backend unavailable — running in simulation mode")
    else:
        st.error(f"🔴 Connection failed: {health.get('detail', 'Unknown error')}")

st.divider()

# ---------------------------------------------------------------------------
# AWS Resource Configuration
# ---------------------------------------------------------------------------

st.markdown("### ☁️ AWS Resources")

config = get_config()

if "error" not in config:
    # Environment info
    env_col1, env_col2 = st.columns(2)
    with env_col1:
        st.markdown(f"**Environment:** `{config.get('environment', 'development')}`")
        st.markdown(f"**Region:** `{config.get('region', 'us-east-1')}`")
    with env_col2:
        st.markdown(f"**API Gateway:** `{config.get('api_gateway_url', 'N/A')}`")
        st.markdown(f"**S3 Bucket:** `{config.get('s3_bucket', 'N/A')}`")

    # Lambda functions
    st.markdown("#### 🔷 Lambda Functions")
    lambda_fns = config.get("lambda_functions", {})
    if lambda_fns:
        for fn_name, fn_config in lambda_fns.items():
            with st.expander(f"📋 {fn_name.title()}", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Memory:** {fn_config.get('memory', 'N/A')} MB")
                with c2:
                    st.markdown(f"**Timeout:** {fn_config.get('timeout', 'N/A')}s")

    # SQS Queues
    st.markdown("#### 📨 SQS Queues")
    sqs_queues = config.get("sqs_queues", {})
    if sqs_queues:
        for queue_name, queue_config in sqs_queues.items():
            with st.expander(f"📬 {queue_name}", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Visibility Timeout:** {queue_config.get('visibility_timeout', 'N/A')}s")
                with c2:
                    st.markdown(f"**Max Receive Count:** {queue_config.get('max_receive_count', 'N/A')}")

    # SNS Topics
    st.markdown("#### 📢 SNS Topics")
    sns_topics = config.get("sns_topics", [])
    if sns_topics:
        for topic in sns_topics:
            st.markdown(f"- `{topic}`")

    # DynamoDB Tables
    st.markdown("#### 🗄️ DynamoDB Tables")
    ddb_tables = config.get("dynamodb_tables", [])
    if ddb_tables:
        for table in ddb_tables:
            st.markdown(f"- `{table}`")

    # Rate Limits
    st.markdown("#### ⏱️ Rate Limits")
    rate_limits = config.get("rate_limits", {})
    if rate_limits:
        rl_col1, rl_col2 = st.columns(2)
        with rl_col1:
            st.markdown(f"**Requests/sec:** {rate_limits.get('requests_per_second', 'N/A')}")
        with rl_col2:
            st.markdown(f"**Burst Limit:** {rate_limits.get('burst_limit', 'N/A')}")
else:
    st.warning("Unable to load configuration. Backend may be unavailable.")

st.divider()

# ---------------------------------------------------------------------------
# Environment Variables Reference
# ---------------------------------------------------------------------------

st.markdown("### 📋 Environment Variables Reference")

env_vars = [
    ("ANIME_QUOTE_API_URL", "FastAPI backend URL", "http://localhost:8000/api/v1"),
    ("ANIME_QUOTE_API_KEY", "API key for authentication", ""),
    ("ANIME_QUOTE_REQUEST_TIMEOUT", "Request timeout in seconds", "30"),
    ("AWS_REGION", "AWS deployment region", "us-east-1"),
    ("S3_BUCKET", "S3 bucket for outputs", "anime-quote-generator-dev"),
    ("DYNAMODB_JOBS_TABLE", "DynamoDB jobs table", "anime-quote-jobs"),
    ("SQS_GENERATION_QUEUE", "SQS generation queue name", "generation-queue"),
    ("SNS_GENERATION_COMPLETE", "SNS success topic", "generation-complete"),
]

env_data = [{"Variable": v[0], "Description": v[1], "Default": v[2]} for v in env_vars]
st.dataframe(env_data, use_container_width=True, hide_index=True)
