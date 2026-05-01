"""
Visualization utilities for the Anime Quote Generator Streamlit dashboard.

Provides reusable chart builders and styling helpers using Plotly
and Streamlit native components.
"""

from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#6366f1",       # Indigo
    "secondary": "#8b5cf6",     # Violet
    "success": "#22c55e",       # Green
    "warning": "#f59e0b",       # Amber
    "danger": "#ef4444",        # Red
    "info": "#3b82f6",          # Blue
    "dark": "#1e293b",          # Slate 800
    "light": "#f1f5f9",         # Slate 100
    "gemini": "#4285f4",        # Google Blue
    "gpt2": "#10b981",          # Emerald
    "fallback": "#f97316",      # Orange
}

SPEECH_TYPE_COLORS = {
    "motivational": "#6366f1",
    "villain": "#ef4444",
    "philosophical": "#8b5cf6",
    "heroic": "#3b82f6",
    "emotional": "#ec4899",
    "comedic": "#f59e0b",
    "dramatic": "#14b8a6",
    "narrative": "#64748b",
}

STATUS_COLORS = {
    "pending": "#f59e0b",
    "preprocessing": "#3b82f6",
    "generating": "#8b5cf6",
    "postprocessing": "#6366f1",
    "completed": "#22c55e",
    "failed": "#ef4444",
    "dlq": "#dc2626",
    "timeout": "#f97316",
    "cancelled": "#64748b",
}


# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------

def render_metric_card(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
) -> None:
    """Render a Streamlit metric card with optional delta indicator."""
    st.metric(
        label=label,
        value=str(value),
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )


def render_metric_row(metrics: List[Dict[str, Any]], columns: int = 4) -> None:
    """Render a row of metric cards.

    Each dict should have keys: label, value, and optionally delta, delta_color, help_text.
    """
    cols = st.columns(min(columns, len(metrics)))
    for idx, metric in enumerate(metrics):
        with cols[idx % columns]:
            render_metric_card(
                label=metric.get("label", ""),
                value=metric.get("value", "—"),
                delta=metric.get("delta"),
                delta_color=metric.get("delta_color", "normal"),
                help_text=metric.get("help_text"),
            )


# ---------------------------------------------------------------------------
# Status indicators
# ---------------------------------------------------------------------------

def status_badge(status: str) -> str:
    """Return an HTML status badge with appropriate color."""
    color = STATUS_COLORS.get(status, "#64748b")
    return (
        f'<span style="background-color:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:0.75rem;font-weight:600;">{status.upper()}</span>'
    )


def render_status_timeline(statuses: List[Dict[str, Any]]) -> None:
    """Render a vertical timeline of job status transitions."""
    for entry in statuses:
        status = entry.get("status", "unknown")
        color = STATUS_COLORS.get(status, "#64748b")
        timestamp = entry.get("timestamp", "")
        detail = entry.get("detail", "")
        st.markdown(
            f'<div style="border-left:3px solid {color};padding-left:12px;margin:4px 0;">'
            f'<span style="color:{color};font-weight:600;">{status}</span>'
            f' <span style="color:#94a3b8;font-size:0.8rem;">{timestamp}</span>'
            f'<br/><span style="font-size:0.85rem;">{detail}</span></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_donut_chart(
    labels: List[str],
    values: List[float],
    title: str = "",
    colors: Optional[List[str]] = None,
) -> go.Figure:
    """Build a donut (ring) chart."""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(
                    colors=colors or [SPEECH_TYPE_COLORS.get(l, COLORS["primary"]) for l in labels],
                    line=dict(color="white", width=2),
                ),
                textinfo="label+percent",
                textfont=dict(size=12),
                hovertemplate="<b>%{label}</b><br>%{value} (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=50, b=50, l=20, r=20),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_bar_chart(
    x: List[Any],
    y: List[float],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: str = COLORS["primary"],
    horizontal: bool = False,
) -> go.Figure:
    """Build a bar chart (vertical or horizontal)."""
    orientation = "h" if horizontal else "v"
    fig = go.Figure(
        data=[
            go.Bar(
                x=y if horizontal else x,
                y=x if horizontal else y,
                orientation=orientation,
                marker=dict(
                    color=color,
                    line=dict(color="white", width=1),
                    cornerradius=4,
                ),
                hovertemplate="<b>%{x if horizontal else y}</b>: %{y if horizontal else x}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title=x_label if not horizontal else y_label,
        yaxis_title=y_label if not horizontal else x_label,
        margin=dict(t=50, b=50, l=50, r=20),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.3,
    )
    return fig


def build_line_chart(
    x: List[Any],
    y: List[float],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: str = COLORS["primary"],
    fill: bool = False,
    secondary_series: Optional[Tuple[List[Any], List[float], str]] = None,
) -> go.Figure:
    """Build a line chart with optional area fill and secondary series."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="Primary",
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
            fill="tozeroy" if fill else None,
            fillcolor=f"rgba(99,102,241,0.1)" if fill else None,
            hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
        )
    )

    if secondary_series:
        sx, sy, sname = secondary_series
        fig.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="lines+markers",
                name=sname,
                line=dict(color=COLORS["secondary"], width=2, dash="dash"),
                marker=dict(size=4),
                hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(t=50, b=50, l=50, r=20),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_gauge_chart(
    value: float,
    title: str = "",
    max_value: float = 100,
    threshold_good: float = 80,
    threshold_warn: float = 50,
) -> go.Figure:
    """Build a gauge (speedometer) chart for KPIs."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text=title, font=dict(size=16)),
            gauge=dict(
                axis=dict(range=[0, max_value], tickwidth=1, tickcolor="white"),
                bar=dict(color=COLORS["primary"], thickness=0.8),
                bgcolor="rgba(0,0,0,0)",
                steps=[
                    dict(range=[0, threshold_warn], color="rgba(239,68,68,0.15)"),
                    dict(range=[threshold_warn, threshold_good], color="rgba(245,158,11,0.15)"),
                    dict(range=[threshold_good, max_value], color="rgba(34,197,94,0.15)"),
                ],
                threshold=dict(
                    line=dict(color=COLORS["danger"], width=3),
                    thickness=0.75,
                    value=value,
                ),
            ),
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(t=60, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_stacked_bar_chart(
    categories: List[str],
    series: Dict[str, List[float]],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> go.Figure:
    """Build a stacked bar chart from multiple series."""
    fig = go.Figure()
    color_list = [COLORS["primary"], COLORS["success"], COLORS["warning"], COLORS["danger"], COLORS["info"]]
    for idx, (name, values) in enumerate(series.items()):
        fig.add_trace(
            go.Bar(
                name=name,
                x=categories,
                y=values,
                marker=dict(color=color_list[idx % len(color_list)], cornerradius=3),
                hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(t=50, b=50, l=50, r=20),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def build_heatmap(
    z: List[List[float]],
    x: List[str],
    y: List[str],
    title: str = "",
    colorscale: str = "Viridis",
) -> go.Figure:
    """Build a heatmap chart."""
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            hovertemplate="%{y} at %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        margin=dict(t=50, b=50, l=80, r=20),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Quote display helpers
# ---------------------------------------------------------------------------

def render_quote_card(
    quote_text: str,
    speech_type: str = "",
    character: str = "",
    source: str = "",
    created_at: str = "",
) -> None:
    """Render a styled quote card."""
    color = SPEECH_TYPE_COLORS.get(speech_type, COLORS["primary"])
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {color};
            background: linear-gradient(135deg, rgba(99,102,241,0.05), rgba(139,92,246,0.05));
            padding: 16px 20px;
            border-radius: 0 8px 8px 0;
            margin: 8px 0;
        ">
            <p style="font-size:1.05rem;font-style:italic;color:#1e293b;margin:0 0 8px 0;">
                "{quote_text}"
            </p>
            <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
                {f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:0.7rem;font-weight:600;">{speech_type.upper()}</span>' if speech_type else ''}
                {f'<span style="color:#64748b;font-size:0.8rem;">🎭 {character}</span>' if character else ''}
                {f'<span style="color:#94a3b8;font-size:0.75rem;">⚡ {source}</span>' if source else ''}
                {f'<span style="color:#94a3b8;font-size:0.75rem;">📅 {created_at}</span>' if created_at else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cost_breakdown(cost_data: Dict[str, float]) -> go.Figure:
    """Build a cost breakdown donut chart."""
    labels = list(cost_data.keys())
    values = list(cost_data.values())
    cost_colors = [
        COLORS["primary"],
        COLORS["success"],
        COLORS["warning"],
        COLORS["info"],
        COLORS["secondary"],
        COLORS["danger"],
    ]
    return build_donut_chart(
        labels=[l.replace("_", " ").title() for l in labels],
        values=values,
        title="Cost Breakdown (USD)",
        colors=cost_colors[: len(labels)],
    )


# ---------------------------------------------------------------------------
# Page layout helpers
# ---------------------------------------------------------------------------

def page_header(title: str, icon: str = "", description: str = "") -> None:
    """Render a consistent page header."""
    st.markdown(
        f'<h1 style="display:flex;align-items:center;gap:8px;">'
        f'{icon} {title}</h1>',
        unsafe_allow_html=True,
    )
    if description:
        st.markdown(f'<p style="color:#64748b;margin-top:-8px;">{description}</p>', unsafe_allow_html=True)
    st.divider()


def section_header(title: str, icon: str = "") -> None:
    """Render a section sub-header."""
    st.markdown(f"### {icon} {title}" if icon else f"### {title}")


def empty_state(message: str, icon: str = "📭") -> None:
    """Render an empty state placeholder."""
    st.markdown(
        f'<div style="text-align:center;padding:40px;color:#94a3b8;">'
        f'<div style="font-size:3rem;">{icon}</div>'
        f'<p style="font-size:1.1rem;">{message}</p></div>',
        unsafe_allow_html=True,
    )
