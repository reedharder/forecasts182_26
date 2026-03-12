# =============================================================================
#  ISO-NE Load Forecast Dashboard  —  Streamlit App
# =============================================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISO-NE Load Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }
  .stApp { background-color: #0d1117; color: #e6edf3; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
  }
  [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

  /* Metric cards */
  .metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 8px;
  }
  .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 4px;
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: #58a6ff;
  }
  .metric-sub {
    font-size: 12px;
    color: #6e7681;
    margin-top: 2px;
  }

  /* Section headers */
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8b949e;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
  }

  /* Tables */
  .dataframe { font-family: 'IBM Plex Mono', monospace; font-size: 13px; }
  thead tr th {
    background-color: #161b22 !important;
    color: #8b949e !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  tbody tr td { background-color: #0d1117 !important; color: #e6edf3 !important; }
  tbody tr:hover td { background-color: #161b22 !important; }

  /* Status badges */
  .badge-ok   { background:#1a3a2a; color:#3fb950; border:1px solid #238636;
                border-radius:4px; padding:2px 8px; font-size:12px; font-family:'IBM Plex Mono',monospace; }
  .badge-warn { background:#2d1f00; color:#e3b341; border:1px solid #9e6a03;
                border-radius:4px; padding:2px 8px; font-size:12px; font-family:'IBM Plex Mono',monospace; }
  .badge-err  { background:#2d0f0f; color:#f85149; border:1px solid #da3633;
                border-radius:4px; padding:2px 8px; font-size:12px; font-family:'IBM Plex Mono',monospace; }

  /* Hide Streamlit chrome */
  #MainMenu {visibility:hidden;} footer {visibility:hidden;}
  [data-testid="stToolbar"] {visibility:hidden;}

  div[data-testid="stExpander"] {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FORECAST_PATH = DATA_DIR / "forecasts.csv"
ACTUAL_PATH   = DATA_DIR / "isone_actual.csv"

ZONE_MAP = {".Z.VERMONT": "VT", ".Z.NEWHAMPSHIRE": "NH"}
STATE_LABELS = {"NH": "New Hampshire", "VT": "Vermont"}
HOURS = list(range(24))

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9", size=11),
    legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1),
    margin=dict(l=50, r=20, t=50, b=50),
    hoverlabel=dict(bgcolor="#161b22", bordercolor="#21262d",
                    font=dict(family="IBM Plex Mono, monospace", size=12)),
)
# Reusable axis style — merged per-chart to avoid duplicate-kwarg collisions
AXIS_STYLE = dict(gridcolor="#21262d", zerolinecolor="#21262d", tickcolor="#8b949e")

# ── Data helpers ──────────────────────────────────────────────────────────────

def file_mtime(path: Path) -> str:
    if path.exists():
        ts = datetime.fromtimestamp(path.stat().st_mtime)
        return ts.strftime("%b %d, %Y  %H:%M:%S")
    return "Not found"

# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_actual(mtime: float) -> tuple[pd.Series, pd.Series, dict]:
    """Load ISO-NE actual file; keyed on mtime so it reloads when file changes."""
    df = pd.read_csv(ACTUAL_PATH, skiprows=[0, 1, 2, 3, 5])
    df.columns = df.columns.str.strip()

    load_col = next(
        c for c in df.columns if "native" in c.lower() and "load" in c.lower()
    )
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")
    df = df.dropna(subset=["Date/Time"])

    df = df[df["Load Zone Name"].isin(ZONE_MAP)].copy()
    df["State"]   = df["Load Zone Name"].map(ZONE_MAP)
    df["Load_MW"] = pd.to_numeric(df[load_col], errors="coerce")
    df["Hour"]    = df["Date/Time"].dt.hour

    counts      = df.groupby(["State", "Hour"])["Load_MW"].count()
    actual_avg  = df.groupby(["State", "Hour"])["Load_MW"].mean()

    last_complete = {}
    for state in ["VT", "NH"]:
        ch = counts.xs(state, level="State")
        ch = ch[ch >= 12]
        last_complete[state] = int(ch.index.max()) if len(ch) else None

    return actual_avg, counts, last_complete


@st.cache_data(show_spinner=False)
def load_forecasts(mtime: float) -> tuple[pd.DataFrame, list]:
    df = pd.read_csv(FORECAST_PATH)
    df.columns = df.columns.str.strip()
    hour_col_map = {c: int(c) for c in df.columns if str(c).strip() in map(str, HOURS)}
    df = df.rename(columns=hour_col_map)
    for h in HOURS:
        if h in df.columns:
            df[h] = pd.to_numeric(
                df[h].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
                errors="coerce",
            )
    df["State"] = df["State/Hour"].astype(str).str.strip().str.upper()
    groups = sorted(df["Group"].unique())
    return df, groups

# ── Analysis helpers ──────────────────────────────────────────────────────────

def get_fc(df_fc, state, group):
    row = df_fc[(df_fc["State"] == state) & (df_fc["Group"] == group)]
    if row.empty:
        return np.full(24, np.nan)
    return row[HOURS].values[0].astype(float)

def avg_forecast(df_fc, groups, state):
    return np.nanmean([get_fc(df_fc, state, g) for g in groups], axis=0)

def get_actuals_array(actual_avg, state):
    out = np.full(24, np.nan)
    for h in HOURS:
        try:
            out[h] = actual_avg.loc[(state, h)]
        except KeyError:
            pass
    return out

def score(fc, act, hours_subset):
    f = fc[hours_subset]
    a = act[hours_subset]
    mask = ~np.isnan(f) & ~np.isnan(a) & (a != 0)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
    diff = f[mask] - a[mask]
    return (
        float(np.mean(np.abs(diff))),
        float(np.mean(np.abs(diff / a[mask])) * 100),
        float(np.sqrt(np.mean(diff ** 2))),
    )

def cumulative_score(fc, act, up_to_hour, metric):
    hrs = list(range(up_to_hour + 1))
    f, a = fc[hrs], act[hrs]
    mask = ~np.isnan(f) & ~np.isnan(a)
    if mask.sum() == 0:
        return np.nan
    diff = f[mask] - a[mask]
    return (float(np.mean(np.abs(diff)))    if metric == "MAE"
            else float(np.sqrt(np.mean(diff**2))))

# ── Plotly helpers ────────────────────────────────────────────────────────────

GROUP_PALETTE = [
    "#58a6ff","#3fb950","#f78166","#d2a8ff","#ffa657",
    "#79c0ff","#56d364","#ff7b72","#bc8cff","#ffc680",
    "#a5d6ff","#85e89d","#ffab70","#e2c9ff","#ffdfb6",
    "#cae8ff","#b4f1b4","#ffd1cc","#eddeff","#fff1d6",
]

def make_ts_figure(state, df_fc, groups, actual_avg, last_complete):
    act = get_actuals_array(actual_avg, state)
    avg = avg_forecast(df_fc, groups, state)
    lc  = last_complete[state]

    fig = go.Figure()

    # Individual group traces
    for i, g in enumerate(groups):
        fc = get_fc(df_fc, state, g)
        fig.add_trace(go.Scatter(
            x=HOURS, y=fc,
            mode="lines",
            name=f"Group {g}",
            line=dict(color=GROUP_PALETTE[i % len(GROUP_PALETTE)], width=1.2),
            opacity=0.45,
            legendgroup=f"group_{g}",
            hovertemplate=f"<b>Group {g}</b><br>Hour: %{{x}}<br>Forecast: %{{y:.1f}} MW<extra></extra>",
        ))

    # Avg forecast
    fig.add_trace(go.Scatter(
        x=HOURS, y=avg,
        mode="lines+markers",
        name="⌀ Avg Forecast",
        line=dict(color="#ffa657", width=3, dash="dot"),
        marker=dict(size=5, color="#ffa657"),
        hovertemplate="<b>Avg Forecast</b><br>Hour: %{x}<br>%{y:.1f} MW<extra></extra>",
        zorder=10,
    ))

    # Actual load
    avail = [(h, act[h]) for h in HOURS if not np.isnan(act[h])]
    if avail:
        xv, yv = zip(*avail)
        fig.add_trace(go.Scatter(
            x=xv, y=yv,
            mode="lines+markers",
            name="Actual Load",
            line=dict(color="#f85149", width=3),
            marker=dict(size=7, color="#f85149", symbol="circle"),
            hovertemplate="<b>Actual</b><br>Hour: %{x}<br>%{y:.1f} MW<extra></extra>",
            zorder=11,
        ))

    subtitle = f"Data through end of hour {lc}" if lc is not None else "No complete hours yet"
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"{STATE_LABELS[state]}  ·  <span style='color:#8b949e;font-size:13px'>{subtitle}</span>",
            font=dict(size=15, color="#e6edf3"),
        ),
        xaxis=dict(**AXIS_STYLE, title="Hour of Day", tickvals=HOURS,
                   ticktext=[str(h) for h in HOURS]),
        yaxis=dict(**AXIS_STYLE, title="Avg Load (MW)"),
        height=380,
        legend=dict(**PLOTLY_LAYOUT["legend"], orientation="v",
                    x=1.01, y=1, xanchor="left"),
        hovermode="x unified",
    )
    return fig


def make_cumulative_figure(state, metric, df_fc, groups, actual_avg, last_complete):
    lc = last_complete[state]
    if lc is None or lc < 1:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT,
                          title=f"{state} — insufficient data for {metric}")
        return fig

    eval_hours = list(range(1, lc + 1))
    act = get_actuals_array(actual_avg, state)

    all_scores = {}
    for g in groups:
        fc = get_fc(df_fc, state, g)
        all_scores[str(g)] = [cumulative_score(fc, act, h, metric) for h in eval_hours]
    avg = avg_forecast(df_fc, groups, state)
    all_scores["Avg Forecast"] = [cumulative_score(avg, act, h, metric) for h in eval_hours]

    # Find ever-top-5
    ever_top5 = {"Avg Forecast"}
    for i in range(len(eval_hours)):
        gs = {g: all_scores[g][i] for g in map(str, groups) if not np.isnan(all_scores[g][i])}
        ever_top5.update(sorted(gs, key=gs.get)[:5])

    fig = go.Figure()
    sorted_groups = sorted(ever_top5, key=lambda x: (x == "Avg Forecast", x))
    for i, g in enumerate(sorted_groups):
        is_avg = g == "Avg Forecast"
        fig.add_trace(go.Scatter(
            x=eval_hours,
            y=all_scores[g],
            mode="lines+markers",
            name=g,
            line=dict(
                color="#ffa657" if is_avg else GROUP_PALETTE[i % len(GROUP_PALETTE)],
                width=3 if is_avg else 2,
                dash="dot" if is_avg else "solid",
            ),
            marker=dict(size=6),
            hovertemplate=f"<b>{g}</b><br>Through hour %{{x}}<br>{metric}: %{{y:.2f}} MW<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"{STATE_LABELS[state]}  ·  Cumulative {metric}",
            font=dict(size=14, color="#e6edf3"),
        ),
        xaxis=dict(**AXIS_STYLE, title="Hour (cumulative through)", tickvals=eval_hours),
        yaxis=dict(**AXIS_STYLE, title=f"{metric} (MW)"),
        height=360,
        legend=dict(**PLOTLY_LAYOUT["legend"], x=1.01, y=1, xanchor="left"),
    )
    return fig


def build_metrics_df(state, df_fc, groups, actual_avg, last_complete):
    lc = last_complete[state]
    if lc is None:
        return None
    hrs = list(range(lc + 1))
    act = get_actuals_array(actual_avg, state)
    rows = []
    for g in groups:
        fc = get_fc(df_fc, state, g)
        mae, mape, rmse = score(fc, act, hrs)
        rows.append({"Group": str(g), "MAE": mae, "MAPE (%)": mape, "RMSE": rmse})
    mae, mape, rmse = score(avg_forecast(df_fc, groups, state), act, hrs)
    rows.append({"Group": "⌀ Avg Forecast", "MAE": mae, "MAPE (%)": mape, "RMSE": rmse})
    return pd.DataFrame(rows).set_index("Group")

# =============================================================================
#  SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
        <div style='padding:16px 0 8px 0'>
          <span style='font-family:IBM Plex Mono,monospace;font-size:18px;
                       font-weight:600;color:#58a6ff;letter-spacing:0.05em'>⚡ ISO-NE</span><br>
          <span style='font-family:IBM Plex Sans,sans-serif;font-size:12px;
                       color:#8b949e;letter-spacing:0.08em;text-transform:uppercase'>
            Forecast Dashboard
          </span>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Forecast file status ──────────────────────────────────────────────────
    st.markdown("**📋 Forecast File**")
    fc_status = "badge-ok" if FORECAST_PATH.exists() else "badge-err"
    fc_label  = "Loaded" if FORECAST_PATH.exists() else "Missing"
    st.markdown(
        f'<span class="{fc_status}">{fc_label}</span>  '
        f'<span style="font-size:11px;color:#6e7681">{file_mtime(FORECAST_PATH)}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:11px;color:#6e7681;margin-top:4px">'
        'Commit <code style="color:#8b949e">data/forecasts.csv</code> to the repo.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Actuals file status ───────────────────────────────────────────────────
    st.markdown("**📡 ISO-NE Actuals**")
    act_status = "badge-ok" if ACTUAL_PATH.exists() else "badge-err"
    act_label  = "Loaded" if ACTUAL_PATH.exists() else "Missing"
    st.markdown(
        f'<span class="{act_status}">{act_label}</span>  '
        f'<span style="font-size:11px;color:#6e7681">{file_mtime(ACTUAL_PATH)}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:11px;color:#6e7681;margin-top:4px;line-height:1.7">'
        'Replace <code style="color:#8b949e">data/isone_actual.csv</code> in the repo each hour.<br>'
        '💡 Download from '
        '<a href="https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/five-min-zone-load" '
        'target="_blank" style="color:#58a6ff">ISO-NE Load Reports</a>, '
        'rename, and push.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        f'<div style="font-size:11px;color:#6e7681">'
        f'Last render: {datetime.now().strftime("%H:%M:%S")}</div>',
        unsafe_allow_html=True,
    )
    if st.button("🔄 Force Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# =============================================================================
#  MAIN CONTENT
# =============================================================================

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:8px 0 24px 0'>
  <h1 style='font-family:IBM Plex Mono,monospace;font-size:28px;font-weight:600;
             color:#e6edf3;margin:0;letter-spacing:-0.01em'>
    ISO-NE Zonal Load Forecasts
  </h1>
  <p style='font-family:IBM Plex Sans,sans-serif;color:#8b949e;margin:6px 0 0 0;font-size:14px'>
    NH · VT  &nbsp;·&nbsp; March 12, 2026  &nbsp;·&nbsp; Five-minute actuals vs hourly group forecasts
  </p>
</div>
""", unsafe_allow_html=True)

# ── Guard: both files must exist ─────────────────────────────────────────────
if not FORECAST_PATH.exists() or not ACTUAL_PATH.exists():
    missing = []
    if not FORECAST_PATH.exists(): missing.append("Forecasts CSV")
    if not ACTUAL_PATH.exists():   missing.append("ISO-NE Actuals CSV")
    st.warning(f"⚠️  Missing files in the repo's `data/` folder: **{', '.join(missing)}**")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
actual_mtime   = ACTUAL_PATH.stat().st_mtime
forecast_mtime = FORECAST_PATH.stat().st_mtime

with st.spinner("Loading data…"):
    actual_avg, counts, last_complete = load_actual(actual_mtime)
    df_fc, groups = load_forecasts(forecast_mtime)

# ── Summary cards ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    lc_nh = last_complete.get("NH")
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">NH Last Complete Hour</div>
      <div class="metric-value">{lc_nh if lc_nh is not None else "—"}</div>
      <div class="metric-sub">{"Hour " + str(lc_nh) + " · 00:00–59:59" if lc_nh is not None else "No complete hours"}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    lc_vt = last_complete.get("VT")
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">VT Last Complete Hour</div>
      <div class="metric-value">{lc_vt if lc_vt is not None else "—"}</div>
      <div class="metric-sub">{"Hour " + str(lc_vt) + " · 00:00–59:59" if lc_vt is not None else "No complete hours"}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Forecast Groups</div>
      <div class="metric-value">{len(groups)}</div>
      <div class="metric-sub">Groups: {", ".join(str(g) for g in groups[:6])}{"…" if len(groups) > 6 else ""}</div>
    </div>""", unsafe_allow_html=True)
with col4:
    total_readings = int(counts.sum())
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">5-min Readings Loaded</div>
      <div class="metric-value">{total_readings:,}</div>
      <div class="metric-sub">NH + VT combined, today</div>
    </div>""", unsafe_allow_html=True)

# ── Time series ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Time Series — Forecasts vs Actuals</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div style="font-size:12px;color:#6e7681;margin-bottom:12px">'
    '🖱️ Hover over any line to see group name and value. '
    'Click legend items to show/hide. Drag to zoom, double-click to reset.'
    '</div>',
    unsafe_allow_html=True,
)

ts_col1, ts_col2 = st.columns(2)
with ts_col1:
    fig_nh = make_ts_figure("NH", df_fc, groups, actual_avg, last_complete)
    st.plotly_chart(fig_nh, use_container_width=True, config={"displayModeBar": False})
with ts_col2:
    fig_vt = make_ts_figure("VT", df_fc, groups, actual_avg, last_complete)
    st.plotly_chart(fig_vt, use_container_width=True, config={"displayModeBar": False})

# ── Metrics tables ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Scoring — All Complete Hours to Date</div>',
            unsafe_allow_html=True)

metric_tabs = st.tabs(["📊 MAE", "📊 MAPE (%)", "📊 RMSE"])
metric_names = ["MAE", "MAPE (%)", "RMSE"]

for tab, metric in zip(metric_tabs, metric_names):
    with tab:
        mc1, mc2 = st.columns(2)
        for col, state in zip([mc1, mc2], ["NH", "VT"]):
            with col:
                lc = last_complete[state]
                st.markdown(
                    f"**{STATE_LABELS[state]}**"
                    + (f"  <span style='font-size:12px;color:#8b949e'>"
                       f"(hours 0–{lc})</span>" if lc is not None else ""),
                    unsafe_allow_html=True,
                )
                df_m = build_metrics_df(state, df_fc, groups, actual_avg, last_complete)
                if df_m is None:
                    st.info("No complete hours available yet.")
                    continue

                top5 = df_m[df_m.index != "⌀ Avg Forecast"].nsmallest(5, metric)
                avg_row = df_m.loc[["⌀ Avg Forecast"]]
                out = pd.concat([top5, avg_row])[[metric]].round(3)

                # Colour the top row gold
                def style_table(df):
                    styles = []
                    for i, idx in enumerate(df.index):
                        if idx == "⌀ Avg Forecast":
                            styles.append("background-color:#2d1f00;color:#ffa657")
                        elif i == 0:
                            styles.append("background-color:#1a2a1a;color:#3fb950;font-weight:600")
                        else:
                            styles.append("")
                    return styles

                st.dataframe(
                    out.style
                       .apply(lambda _: style_table(out), axis=0)
                       .format("{:.3f}")
                       .set_properties(**{"text-align": "right"}),
                    use_container_width=True,
                    height=min(250, (len(out) + 1) * 38),
                )

# ── Cumulative plots ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Cumulative Scores Over Time</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div style="font-size:12px;color:#6e7681;margin-bottom:12px">'
    'Only groups that appeared in the top 5 at any hour are shown.'
    '</div>',
    unsafe_allow_html=True,
)

cum_tabs = st.tabs(["📈 Cumulative MAE", "📈 Cumulative RMSE"])
for tab, metric in zip(cum_tabs, ["MAE", "RMSE"]):
    with tab:
        cc1, cc2 = st.columns(2)
        for col, state in zip([cc1, cc2], ["NH", "VT"]):
            with col:
                fig = make_cumulative_figure(
                    state, metric, df_fc, groups, actual_avg, last_complete
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
