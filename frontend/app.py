"""
Anime Quote Generator — Streamlit Dashboard

Main entry point for the multi-page Streamlit application.
Handles authentication, session initialization, and sidebar navigation.
"""

import os
import sys

import streamlit as st

# Ensure the project root is on the path so utils can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.aws_client import check_health, login, logout
from utils.visualization import page_header, COLORS

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Anime Quote Generator",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Global */
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99,102,241,0.3);
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 8px;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Hide default page navigation (we use sidebar) */
    [data-testid="stSidebarNav"] { display: none; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize default session state values."""
    defaults = {
        "authenticated": False,
        "auth_token": None,
        "refresh_token": None,
        "user_info": {},
        "api_base_url": os.getenv("ANIME_QUOTE_API_URL", "http://localhost:8000/api/v1"),
        "api_key": os.getenv("ANIME_QUOTE_API_KEY", ""),
        "simulation_mode": False,
        "recent_jobs": [],
        "sidebar_page": "🏠 Dashboard",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Logo / Title
    st.markdown(
        """
        <div style="text-align:center;padding:16px 0 8px 0;">
            <div style="font-size:2.5rem;">🎭</div>
            <h2 style="margin:0;color:#f1f5f9;">Anime Quote Gen</h2>
            <p style="color:#94a3b8;font-size:0.8rem;margin:4px 0 0 0;">AWS Lambda Pipeline</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Navigation
    pages = [
        "🏠 Dashboard",
        "🎭 Generate",
        "📊 History",
        "⚙️ Configuration",
        "📈 Monitoring",
    ]
    selected_page = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.get("sidebar_page", "🏠 Dashboard")),
        label_visibility="collapsed",
    )
    st.session_state["sidebar_page"] = selected_page

    st.divider()

    # Authentication section
    st.markdown("#### 🔐 Authentication")
    if st.session_state.get("authenticated"):
        user_info = st.session_state.get("user_info", {})
        st.markdown(
            f"""
            <div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);
                        border-radius:8px;padding:10px;margin:4px 0;">
                <div style="color:#22c55e;font-weight:600;">✓ Authenticated</div>
                <div style="color:#94a3b8;font-size:0.85rem;">
                    👤 {user_info.get('username', 'User')}<br/>
                    🏷️ {user_info.get('role', 'user').title()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()
    else:
        with st.expander("Login", expanded=True):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Sign In", use_container_width=True, type="primary"):
                if username and password:
                    result = login(username, password)
                    if "error" in result:
                        st.error(f"Login failed: {result.get('detail', 'Unknown error')}")
                    else:
                        st.success("Logged in successfully!")
                        st.rerun()
                else:
                    st.warning("Please enter both username and password.")

    st.divider()

    # Backend status
    st.markdown("#### 📡 Backend Status")
    if st.button("Check Connection", use_container_width=True):
        health = check_health()
        if health.get("status") == "healthy" or health.get("status") == "ok":
            st.session_state["simulation_mode"] = False
            st.success("🟢 Backend connected")
        elif health.get("status") == "unavailable":
            st.session_state["simulation_mode"] = True
            st.warning("🟡 Backend unavailable — Simulation mode")
        else:
            st.session_state["simulation_mode"] = True
            st.error(f"🔴 Backend error: {health.get('detail', 'Unknown')}")

    mode = "🔌 Connected" if not st.session_state.get("simulation_mode") else "🎮 Simulation"
    st.caption(f"Mode: {mode}")

    # API URL configuration
    with st.expander("🔧 API Settings"):
        new_url = st.text_input(
            "API Base URL",
            value=st.session_state.get("api_base_url", ""),
            key="api_url_input",
        )
        if new_url != st.session_state.get("api_base_url"):
            st.session_state["api_base_url"] = new_url

        new_key = st.text_input(
            "API Key",
            value=st.session_state.get("api_key", ""),
            type="password",
            key="api_key_input",
        )
        if new_key != st.session_state.get("api_key"):
            st.session_state["api_key"] = new_key

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

page_map = {
    "🏠 Dashboard": "pages/01_🏠_Dashboard",
    "🎭 Generate": "pages/02_🎭_Generate",
    "📊 History": "pages/03_📊_History",
    "⚙️ Configuration": "pages/04_⚙️_Configuration",
    "📈 Monitoring": "pages/05_📈_Monitoring",
}

selected = st.session_state.get("sidebar_page", "🏠 Dashboard")
page_module = page_map.get(selected)

if page_module:
    try:
        st.switch_page(f"{page_module}.py")
    except Exception:
        # Fallback: render inline for the dashboard page
        _render_dashboard_fallback()
else:
    _render_dashboard_fallback()


def _render_dashboard_fallback() -> None:
    """Render a simple landing page when multi-page navigation is not available."""
    page_header("Anime Quote Generator", "🎭", "AWS Lambda-powered anime speech generation pipeline")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Generations", "1,247", "+23 today")
    with col2:
        st.metric("Success Rate", "95.3%", "+0.5%")
    with col3:
        st.metric("Avg. Processing", "3.2s", "-0.3s")
    with col4:
        st.metric("Active Jobs", "3", "—")

    st.info("👈 Use the sidebar to navigate between pages. Click **Check Connection** to verify backend status.")
