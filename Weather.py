# Weather.py
# BlendWX — Weather Model Comparison (Open-Meteo)
#
# Run (PowerShell):
#   cd "$HOME\OneDrive - Altas Corporation\Desktop\weather-app"
#   .\.venv\Scripts\Activate.ps1
#   python -m streamlit run Weather.py

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# -----------------------------
# App constants
# -----------------------------
APP_NAME = "BlendWX"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Models supported via /v1/forecast?models=...
MODEL_CANDIDATES = {
    "ECMWF IFS": ["ecmwf_ifs", "ecmwf_ifs04", "ecmwf_ifs025"],
    "GFS Seamless": ["gfs_seamless"],
    "GEM Seamless": ["gem_seamless"],
}
DEFAULT_MODELS = ["ECMWF IFS", "GFS Seamless", "GEM Seamless"]

HORIZON_OPTIONS = [7, 10, 14]
DEFAULT_HORIZON = 10

PRECIP_BAR_INTERVALS = ["1h", "3h", "6h", "12h", "24h"]
DEFAULT_BAR_INTERVAL = "6h"

# Hourly variables supported in /v1/forecast (keep conservative)
HOURLY_REQUEST = {
    "temp_c": "temperature_2m",
    "feels_like_c": "apparent_temperature",
    "rh_pct": "relative_humidity_2m",
    "cloud_pct": "cloud_cover",
    "precip_mm": "precipitation",
    "precip_prob_pct": "precipitation_probability",
    "wind_kmh": "wind_speed_10m",
    "wind_gust_kmh": "wind_gusts_10m",
    "pressure_hpa": "surface_pressure",
    "dni_wm2": "direct_normal_irradiance",
    "sunshine_duration_s": "sunshine_duration",
    "cape_jkg": "cape",
    "weather_code": "weather_code",
}

DNI_SUNSHINE_THRESHOLD_WM2 = 120.0  # WMO definition: DNI > 120 W/m² = sunshine

MODEL_COLORS: dict[str, str] = {
    "Blend":        "#6366f1",   # indigo — matches WX logo
    "ECMWF IFS":    "#3b82f6",   # blue
    "GFS Seamless": "#10b981",   # emerald
    "GEM Seamless": "#f97316",   # orange — distinct from the others
}

# WMO severity ranking — higher = more significant condition for daily summary
WMO_SEVERITY: dict[int, int] = {
    0: 0, 1: 1, 2: 2, 3: 3,
    45: 5, 48: 6,
    51: 7, 53: 8, 55: 9,
    56: 8, 57: 9,
    61: 10, 63: 11, 65: 12,
    66: 11, 67: 12,
    71: 10, 73: 11, 75: 12,
    77: 9,
    80: 10, 81: 11, 82: 12,
    85: 11, 86: 12,
    95: 13, 96: 14, 99: 15,
}

# WMO weather code → display emoji
WMO_EMOJI: dict[int, str] = {
    0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
    45: "🌫️", 48: "🌫️",
    51: "🌦️", 53: "🌦️", 55: "🌦️",
    56: "🌧️", 57: "🌧️",          # freezing drizzle
    61: "🌧️", 63: "🌧️", 65: "🌧️",
    66: "🌧️", 67: "🌧️",          # freezing rain
    71: "🌨️", 73: "🌨️", 75: "🌨️",
    77: "🌨️",                      # snow grains
    80: "🌧️", 81: "🌧️", 82: "🌧️",
    85: "🌨️", 86: "🌨️",
    95: "⛈️", 96: "⛈️", 99: "⛈️",
}

# X-axis formatting: short ticks, detailed hover (North American, AM/PM)
X_TICKFORMAT = "%I %p"  # e.g., 01 PM
X_HOVERFORMAT = "%a %b %d, %I:%M %p"


# -----------------------------
# HTTP session (retries help with intermittent TLS / proxy flakiness)
# -----------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "BlendWX/1.0"})
    return s


SESSION = make_session()


# -----------------------------
# Data utilities
# -----------------------------
@st.cache_data(show_spinner=False, ttl=86400)  # 24 h — coordinates don't change
def geocode_city(name: str, count: int = 10) -> list[dict]:
    name = (name or "").strip()
    if not name:
        return []
    params = {"name": name, "count": count, "language": "en", "format": "json"}
    r = SESSION.get(GEOCODE_URL, params=params, timeout=25)
    r.raise_for_status()
    return (r.json().get("results") or [])[:count]


def build_params(lat: float, lon: float, forecast_days: int, model_slug: str) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_REQUEST.values()),
        "forecast_days": int(forecast_days),
        "timezone": "auto",
        "timeformat": "iso8601",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "wind_speed_unit": "kmh",
        "models": model_slug,
    }


def parse_time_index(times, tz_name: str | None) -> tuple[pd.DatetimeIndex, str]:
    s = pd.Series(pd.to_datetime(times, errors="coerce"), name="timestamp").dropna()

    tz_label = tz_name or "UTC"
    if tz_name:
        try:
            tz = ZoneInfo(tz_name)
            s = s.dt.tz_localize(tz)
        except Exception:
            tz_label = "UTC"
            s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_localize("UTC")

    return pd.DatetimeIndex(s), tz_label


@st.cache_data(show_spinner=False, ttl=3600)  # 1 h — model runs publish every 6 h
def fetch_one_model(display_name: str, model_slug: str, lat: float, lon: float, forecast_days: int) -> pd.DataFrame:
    params = build_params(lat, lon, forecast_days, model_slug)
    r = SESSION.get(FORECAST_URL, params=params, timeout=35)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.reason}: {r.text[:300]}")
    payload = r.json() or {}

    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    tz_name = payload.get("timezone")

    ts, tz_label = parse_time_index(times, tz_name)
    n = len(ts)

    def get_arr(hourly_key: str) -> pd.Series:
        arr = hourly.get(hourly_key)
        if arr is None:
            return pd.Series([pd.NA] * n)
        arr = list(arr)
        if len(arr) >= n:
            arr = arr[:n]
        else:
            arr = arr + [pd.NA] * (n - len(arr))
        return pd.Series(arr)

    df = pd.DataFrame({"timestamp": ts, "model": display_name, "model_slug": model_slug})
    for canon, api_key in HOURLY_REQUEST.items():
        df[canon] = pd.to_numeric(get_arr(api_key), errors="coerce")

    # clip percent variables
    for c in ["rh_pct", "cloud_pct", "precip_prob_pct"]:
        df[c] = df[c].clip(lower=0, upper=100)

    # defensive: precipitation should not be negative
    df["precip_mm"] = df["precip_mm"].where(df["precip_mm"].isna() | (df["precip_mm"] >= 0), np.nan)

    df.attrs["timezone"] = tz_label
    df.attrs["model_slug"] = model_slug
    return df


@st.cache_data(show_spinner=False)
def fetch_model_with_candidates(display_name: str, lat: float, lon: float, forecast_days: int) -> pd.DataFrame:
    candidates = MODEL_CANDIDATES.get(display_name, [])
    if not candidates:
        raise ValueError(f"Model not configured: {display_name}")

    last_err = None
    for slug in candidates:
        try:
            return fetch_one_model(display_name, slug, lat, lon, forecast_days)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All candidate slugs failed. Last error: {last_err}")


def add_blend(df: pd.DataFrame, label: str = "Blend") -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [c for c in df.columns if c not in ("timestamp", "model", "model_slug")]
    b = df.groupby("timestamp", as_index=False)[numeric_cols].mean(numeric_only=True)
    b["model"] = label
    b["model_slug"] = "blend"
    return pd.concat([df, b], ignore_index=True, sort=False)


def compute_daily_sunshine(df_hourly: pd.DataFrame) -> pd.DataFrame:
    if df_hourly.empty:
        return pd.DataFrame(columns=["date", "model", "sunshine_hours", "method"])

    out = df_hourly.copy()
    out["date"] = out["timestamp"].dt.floor("D")

    if out["sunshine_duration_s"].notna().any():
        daily = (
            out.groupby(["date", "model"], as_index=False)["sunshine_duration_s"]
            .sum(min_count=1)
            .rename(columns={"sunshine_duration_s": "sunshine_seconds"})
        )
        daily["sunshine_hours"] = daily["sunshine_seconds"] / 3600.0
        daily["method"] = "Native sunshine duration"
        return daily.drop(columns=["sunshine_seconds"])

    if out["dni_wm2"].notna().any():
        out["sunny_hour"] = out["dni_wm2"] > DNI_SUNSHINE_THRESHOLD_WM2
        daily = (
            out.groupby(["date", "model"], as_index=False)["sunny_hour"]
            .sum(min_count=1)
            .rename(columns={"sunny_hour": "sunshine_hours"})
        )
        daily["sunshine_hours"] = pd.to_numeric(daily["sunshine_hours"], errors="coerce")
        daily["method"] = f"Estimated (DNI > {DNI_SUNSHINE_THRESHOLD_WM2:g} W/m²)"
        return daily

    daily = out.groupby(["date", "model"], as_index=False).size()[["date", "model"]]
    daily["sunshine_hours"] = pd.NA
    daily["method"] = "Not available"
    return daily




# -----------------------------
# Units, badges, headline, daily cards
# -----------------------------
def convert_units(df: pd.DataFrame, imperial: bool) -> pd.DataFrame:
    if not imperial:
        return df
    df = df.copy()
    for col in ("temp_c", "feels_like_c"):
        if col in df.columns:
            df[col] = df[col] * 9 / 5 + 32
    for col in ("wind_kmh", "wind_gust_kmh"):
        if col in df.columns:
            df[col] = df[col] * 0.621371
    if "precip_mm" in df.columns:
        df["precip_mm"] = df["precip_mm"] * 0.0393701
    return df


def model_spread_label(df: pd.DataFrame, col: str, t_yellow: float, t_red: float) -> str:
    """Return a model agreement emoji for the given metric column."""
    non_blend = df[df["model"] != "Blend"][["timestamp", "model", col]].dropna(subset=[col])
    if non_blend["model"].nunique() < 2:
        return ""
    spread = non_blend.groupby("timestamp")[col].std(ddof=1).dropna().mean()
    if pd.isna(spread):
        return ""
    if spread < t_yellow:
        return "🟢"
    if spread < t_red:
        return "🟡"
    return "🔴"


def compute_headline(df_all: pd.DataFrame) -> str:
    """Generate a one-line forecast summary from blend data."""
    blend = df_all[df_all["model"] == "Blend"].copy()
    if blend.empty:
        return ""
    blend["date"] = blend["timestamp"].dt.floor("D")

    if "weather_code" in blend.columns:
        thunder = blend[blend["weather_code"].isin([95, 96, 99])]
        if not thunder.empty:
            day = thunder["timestamp"].min().strftime("%A")
            return f"⛈️  Thunderstorms possible {day}"

    if "precip_prob_pct" in blend.columns:
        daily_pop = blend.groupby("date")["precip_prob_pct"].max()
        rainy = daily_pop[daily_pop >= 60]
        if not rainy.empty:
            day = pd.Timestamp(rainy.index[0]).strftime("%A")
            return f"🌧️  Rain expected {day}"

    non_blend = df_all[df_all["model"] != "Blend"]
    if non_blend["model"].nunique() >= 2:
        spread = non_blend.groupby("timestamp")["temp_c"].std(ddof=1).dropna().mean()
        if not pd.isna(spread) and spread >= 3.0:
            return "🟡  Models are uncertain — check back closer to the date"

    if "weather_code" in blend.columns and blend["weather_code"].notna().any():
        modal_code = int(blend["weather_code"].dropna().mode().iloc[0])
        emoji = WMO_EMOJI.get(modal_code, "🌡️")
        if modal_code == 0:
            return f"{emoji}  Clear skies ahead"
        if modal_code <= 3:
            return f"{emoji}  Partly cloudy conditions"
        if modal_code < 60:
            return f"{emoji}  Some cloud and mist expected"
        if modal_code < 70:
            return f"{emoji}  Rainy conditions ahead"
        if modal_code < 80:
            return f"{emoji}  Snow in the forecast"

    return "📊  Forecast loaded — scroll down to explore"


def _modal_wmo(s: pd.Series) -> int:
    """Return the most common WMO code in the series."""
    s = s.dropna().astype(int)
    return int(s.mode().iloc[0]) if not s.empty else 0


def _dominant_wmo(s: pd.Series) -> int:
    """Return the highest-severity WMO code (industry standard: any occurrence of rain
    beats many hours of clear skies — same logic as Dark Sky / Visual Crossing)."""
    s = s.dropna().astype(int)
    if s.empty:
        return 0
    return max(s.unique(), key=lambda c: WMO_SEVERITY.get(c, 0))


def render_daily_cards(df_all: pd.DataFrame, imperial: bool) -> None:
    """Render a horizontal row of daily forecast summary cards."""
    blend = df_all[df_all["model"] == "Blend"].copy()
    if blend.empty:
        return
    blend["date"] = blend["timestamp"].dt.floor("D")

    daily = blend.groupby("date").agg(
        temp_max=("temp_c", "max"),
        temp_min=("temp_c", "min"),
        precip_prob=("precip_prob_pct", "max"),
        precip_total=("precip_mm", "sum"),
    ).reset_index()

    # Weather code: highest-severity WMO code during daytime hours (7am–7pm).
    # Severity-based = any single hour of rain beats a mostly-clear day (Dark Sky / Visual
    # Crossing approach). Daytime filter prevents overnight showers ruining a sunny day.
    # Falls back to all hours if no daytime data available.
    raw_models = df_all[df_all["model"] != "Blend"].copy()
    if not raw_models.empty and "weather_code" in raw_models.columns:
        raw_models["date"] = raw_models["timestamp"].dt.floor("D")
        daytime = raw_models[raw_models["timestamp"].dt.hour.between(7, 18)]
        src = daytime if not daytime.empty else raw_models
        wcode_by_day = (
            src.groupby("date")["weather_code"]
            .agg(_dominant_wmo)
            .reset_index()
            .rename(columns={"weather_code": "wcode"})
        )
        daily = daily.merge(wcode_by_day, on="date", how="left")
        daily["wcode"] = daily["wcode"].fillna(0).astype(int)
    else:
        daily["wcode"] = 0

    if imperial:
        daily["temp_max"]     = daily["temp_max"] * 9 / 5 + 32
        daily["temp_min"]     = daily["temp_min"] * 9 / 5 + 32
        daily["precip_total"] = daily["precip_total"] * 0.0393701
    t_sym = "°F" if imperial else "°C"
    p_sym = "in" if imperial else "mm"

    # Threshold: only show precip amount if ≥ 0.5mm (avoids "0mm" from tiny trace amounts)
    _ptot_threshold = 0.02 if imperial else 0.5

    cards = []
    for _, row in daily.head(10).iterrows():
        date = pd.Timestamp(row["date"])
        emoji = WMO_EMOJI.get(int(row["wcode"]), "🌦️")
        pop   = f"{row['precip_prob']:.0f}%" if pd.notna(row["precip_prob"]) else "—"
        ptot  = f"{row['precip_total']:.0f}{p_sym}" if pd.notna(row["precip_total"]) and row["precip_total"] >= _ptot_threshold else "—"
        cards.append(f"""
        <div style="min-width:76px;flex:1;padding:10px 6px;border-radius:12px;
                    background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.10);
                    text-align:center;font-size:0.82rem;color:rgba(255,255,255,0.85);">
          <div style="font-weight:600;font-size:0.88rem;">{date.strftime('%a')}</div>
          <div style="font-size:0.72rem;color:rgba(255,255,255,0.45);margin-bottom:4px;">{date.strftime('%b %d')}</div>
          <div style="font-size:1.6rem;line-height:1.2;">{emoji}</div>
          <div style="font-weight:600;margin-top:4px;">{row['temp_max']:.0f}{t_sym}</div>
          <div style="color:rgba(255,255,255,0.50);">{row['temp_min']:.0f}{t_sym}</div>
          <div style="margin-top:5px;font-size:0.72rem;color:rgba(100,160,255,0.85);">💧 {ptot}</div>
          <div style="font-size:0.72rem;color:rgba(150,190,255,0.70);">{pop} chance</div>
        </div>""")

    html = f'<div style="display:flex;gap:8px;overflow-x:auto;padding:4px 0 14px 0;">{"".join(cards)}</div>'
    st.markdown(html, unsafe_allow_html=True)


# -----------------------------
# Plot helpers (dark theme, hover, day separators)
# -----------------------------
def _midnights_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if start.tzinfo is None or end.tzinfo is None:
        return []
    d0 = start.floor("D") + pd.Timedelta(days=1)
    mids = []
    cur = d0
    while cur < end:
        mids.append(cur)
        cur = cur + pd.Timedelta(days=1)
    return mids


def _apply_chart_theme(fig: go.Figure, start: pd.Timestamp, end: pd.Timestamp):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(18,20,26,0.96)",
            bordercolor="rgba(255,255,255,0.12)",
            font=dict(size=13, color="rgba(255,255,255,0.92)"),
        ),
        margin=dict(l=12, r=12, t=44, b=20),
        showlegend=False,
    )

    fig.update_xaxes(
        showgrid=False,           # no vertical grid lines — midnight vlines handle separation
        dtick=43200000,           # tick labels every 12 h ("12 AM" + "12 PM") — no lines
        tickformat=X_TICKFORMAT,
        hoverformat=X_HOVERFORMAT,
        title_text="",
        zeroline=False,
        tickfont=dict(size=12, color="rgba(255,255,255,0.35)"),
        tickcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=13),
        title_font=dict(size=12, color="rgba(255,255,255,0.50)"),
        title_standoff=8,
    )

    # Day separators at midnight
    for m in _midnights_between(start, end):
        fig.add_vline(x=m, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.15)")

    # Uppercase day labels centred at midday
    if start.tzinfo is not None:
        day_start = start.floor("D")
        day_end = end.floor("D")
        cur = day_start
        while cur <= day_end:
            mid = cur + pd.Timedelta(hours=12)
            if start <= mid <= end:
                fig.add_annotation(
                    x=mid,
                    y=1.0,
                    xref="x",
                    yref="paper",
                    yanchor="bottom",
                    text=mid.strftime("%a %b %d").upper(),
                    showarrow=False,
                    font=dict(size=13, color="rgba(255,255,255,0.55)", family="Inter, sans-serif"),
                )
            cur += pd.Timedelta(days=1)

    return fig


def _hovertemplate(name: str, unit: str | None):
    u = f" {unit}" if unit else ""
    return f"<b>%{{fullData.name}}</b><br>{name}: <b>%{{y:.1f}}{u}</b><extra></extra>"


def _model_order(unique_models, display_mode: str) -> list[str]:
    """Return model trace order for a given display mode."""
    has_blend = "Blend" in unique_models
    if display_mode == "Blend only":
        return ["Blend"] if has_blend else []
    if display_mode == "Models only":
        return [m for m in unique_models if m != "Blend"]
    return (["Blend"] if has_blend else []) + [m for m in unique_models if m != "Blend"]


def _trace_style(model_name: str, display_mode: str) -> tuple[dict, float]:
    """Return (line dict, opacity) for a model trace."""
    color = MODEL_COLORS.get(model_name, "#94a3b8")
    if display_mode == "Models + Blend":
        if model_name == "Blend":
            return dict(width=4, color=color), 1.0
        return dict(width=1, color=color), 0.28
    # Models only or Blend only
    return dict(width=4 if model_name == "Blend" else 2, color=color), 1.0


def _chart_panel_header(title: str, df: pd.DataFrame, display_mode: str) -> None:
    """Render a styled glass-card panel header with inline model legend before a chart."""
    models_in_df = df["model"].unique().tolist() if not df.empty else []
    ordered = _model_order(models_in_df, display_mode)

    swatches = ""
    for m in ordered:
        color = MODEL_COLORS.get(m, "#94a3b8")
        swatches += (
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'margin-right:14px;font-size:0.85rem;color:rgba(255,255,255,0.70);">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{color};flex-shrink:0;"></span>{m}</span>'
        )

    st.markdown(
        f'<div class="bwx-panel-hdr">'
        f'<span class="bwx-panel-title">{title}</span>'
        f'<span class="bwx-panel-legend">{swatches}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def plot_timeseries(
    df: pd.DataFrame,
    y: str,
    title: str,
    y_title: str,
    unit: str | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    display_mode: str,
    y_range: list[float | None] | None = None,
    to_zero: bool = False,
    smooth_window: int = 0,
):
    plot_df = df[["timestamp", "model", y]].copy().dropna(subset=["timestamp"])
    if plot_df.empty:
        st.info("No data available for this metric over the selected time window.")
        return

    fig = go.Figure()
    for model_name in _model_order(plot_df["model"].unique(), display_mode):
        g = plot_df[plot_df["model"] == model_name].sort_values("timestamp")
        if g.empty:
            continue

        line, opacity = _trace_style(model_name, display_mode)
        y_vals = g[y].rolling(smooth_window, center=True, min_periods=1).mean() if smooth_window > 1 else g[y]
        fig.add_trace(
            go.Scatter(
                x=g["timestamp"],
                y=y_vals,
                mode="lines",
                name=model_name,
                line=line,
                opacity=opacity,
                hovertemplate=_hovertemplate(y_title, unit),
            )
        )

    fig.update_yaxes(title_text=y_title)

    if to_zero:
        fig.update_yaxes(rangemode="tozero")
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    _apply_chart_theme(fig, start, end)
    st.plotly_chart(fig, width="stretch")


def plot_precip_bars(
    df: pd.DataFrame,
    title: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    display_mode: str,
    bar_interval: str,
    p_unit: str = "mm",
):
    plot_df = df[["timestamp", "model", "precip_mm"]].copy()
    plot_df["precip_mm"] = pd.to_numeric(plot_df["precip_mm"], errors="coerce").fillna(0.0)

    if display_mode == "Blend only":
        plot_df = plot_df[plot_df["model"] == "Blend"]
    elif display_mode == "Models only":
        plot_df = plot_df[plot_df["model"] != "Blend"]

    if plot_df.empty:
        st.info("No precipitation data available for the selected time window.")
        return

    plot_df["bucket"] = plot_df["timestamp"].dt.floor(bar_interval)
    agg = plot_df.groupby(["bucket", "model"], as_index=False)["precip_mm"].sum()

    model_order = _model_order(agg["model"].unique(), display_mode)

    fig = go.Figure()
    for m in model_order:
        g = agg[agg["model"] == m].sort_values("bucket")
        if g.empty:
            continue

        color = MODEL_COLORS.get(m, "#94a3b8")
        if display_mode == "Models + Blend":
            marker = dict(color=color, opacity=0.85 if m == "Blend" else 0.50)
        else:
            marker = dict(color=color, opacity=0.85)

        fig.add_trace(
            go.Bar(
                x=g["bucket"],
                y=g["precip_mm"],
                name=m,
                marker=marker,
                hovertemplate="<b>%{fullData.name}</b><br>Precipitation: <b>%{y:.1f} mm</b><extra></extra>",
            )
        )

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text=f"Precipitation ({p_unit})", rangemode="tozero")
    _apply_chart_theme(fig, start, end)
    st.plotly_chart(fig, width="stretch")


def plot_precip_cumulative(
    df: pd.DataFrame,
    title: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    display_mode: str,
    p_unit: str = "mm",
):
    x = df[["timestamp", "model", "precip_mm"]].copy()
    x["precip_mm"] = pd.to_numeric(x["precip_mm"], errors="coerce").fillna(0.0)
    x = x.sort_values(["model", "timestamp"])
    x["cum_precip_mm"] = x.groupby("model")["precip_mm"].cumsum()

    if display_mode == "Blend only":
        x = x[x["model"] == "Blend"]
    elif display_mode == "Models only":
        x = x[x["model"] != "Blend"]

    if x.empty:
        st.info("No precipitation data available for the selected time window.")
        return

    tmp = x.rename(columns={"cum_precip_mm": "_cum"})
    plot_timeseries(
        df=tmp,
        y="_cum",
        title=title,
        y_title=f"Cumulative precipitation ({p_unit})",
        unit=p_unit,
        start=start,
        end=end,
        display_mode=display_mode,
        y_range=None,
        to_zero=True,
    )


def plot_daily_sunshine(
    daily_sun: pd.DataFrame,
    display_mode: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
):
    if daily_sun.empty or daily_sun["sunshine_hours"].isna().all():
        st.info("Sunshine hours are not available for the selected time window/models.")
        return

    if display_mode == "Blend only":
        daily_sun = daily_sun[daily_sun["model"] == "Blend"]
    elif display_mode == "Models only":
        daily_sun = daily_sun[daily_sun["model"] != "Blend"]

    fig = go.Figure()
    for m in _model_order(daily_sun["model"].unique(), display_mode):
        g = daily_sun[daily_sun["model"] == m].sort_values("date")
        if g.empty:
            continue

        line, opacity = _trace_style(m, display_mode)
        fig.add_trace(
            go.Scatter(
                x=g["date"] + pd.Timedelta(hours=12),
                y=g["sunshine_hours"],
                mode="lines+markers",
                name=m,
                line=line,
                opacity=opacity,
                hovertemplate="<b>%{fullData.name}</b><br>Sunshine: <b>%{y:.1f} h</b><extra></extra>",
            )
        )

    fig.update_yaxes(title_text="Hours", rangemode="tozero")
    fig.update_xaxes(
        title_text="",
        tickformat="%a %b %d",
        hoverformat="%a %b %d",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
    )
    _apply_chart_theme(fig, start_ts, end_ts)
    st.plotly_chart(fig, width="stretch")



# -----------------------------
# Streamlit page theming (dark, modern) + header/instructions
# -----------------------------
st.set_page_config(page_title=APP_NAME, layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

      /* ---- Base ---- */
      .stApp {
        background: radial-gradient(ellipse 120% 60% at 10% 0%, rgba(99,102,241,0.18) 0%, transparent 60%),
                    radial-gradient(ellipse 90% 50% at 90% 5%, rgba(16,185,129,0.10) 0%, transparent 60%),
                    #030712;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
      }
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      h1, h2, h3 { letter-spacing: -0.02em; color: #f8fafc; }

      /* ---- Glass card ---- */
      .bwx-glass {
        background: rgba(30,41,59,0.40);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 14px;
      }

      /* ---- Chart panel header ---- */
      .bwx-panel-hdr {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 6px;
        padding: 10px 14px 8px 14px;
        background: rgba(30,41,59,0.50);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.06);
        border-bottom: none;
        border-radius: 12px 12px 0 0;
        margin-top: 18px;
        margin-bottom: 0;
      }
      .bwx-panel-title {
        font-size: 0.88rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(248,250,252,0.60);
      }
      .bwx-panel-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 0;
      }

      /* ---- Glass pills ---- */
      .bwx-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.30);
        font-size: 0.88rem;
        color: rgba(248,250,252,0.75);
        margin-right: 6px;
        margin-bottom: 4px;
      }

      /* ---- Sidebar ---- */
      section[data-testid="stSidebar"] {
        background: rgba(15,20,35,0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
      }
      .bwx-sidebar-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: rgba(248,250,252,0.35);
        margin: 12px 0 4px 0;
      }
      section[data-testid="stSidebar"] hr {
        margin: 8px 0 !important;
        border-color: rgba(255,255,255,0.07) !important;
      }
      section[data-testid="stSidebar"] .stRadio label { font-size: 0.92rem; }
      section[data-testid="stSidebar"] .stSelectbox label,
      section[data-testid="stSidebar"] .stTextInput label { font-size: 0.92rem; color: rgba(248,250,252,0.65); }
      section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }

      /* ---- Headline / subtle text ---- */
      .bwx-headline { color: rgba(248,250,252,0.72); font-size: 1.0rem; margin: 4px 0 10px 0; }

      /* ---- Footer ---- */
      .bwx-footer {
        margin-top: 24px;
        padding: 14px 0;
        border-top: 1px solid rgba(255,255,255,0.07);
        text-align: center;
        font-size: 0.75rem;
        color: rgba(248,250,252,0.30);
        letter-spacing: 0.02em;
      }
      .bwx-footer a { color: rgba(99,102,241,0.70); text-decoration: none; }

      @media (max-width: 640px) {
        .bwx-panel-hdr { padding: 8px 10px 6px; }
        .bwx-pill { font-size: 0.72rem; padding: 2px 8px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    # Logo
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;padding:6px 0 12px 0;">'
        '<div style="width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#6366f1,#8b5cf6);'
        'display:flex;align-items:center;justify-content:center;font-size:1.1rem;">⛅</div>'
        '<span style="font-size:1.25rem;font-weight:700;color:#f8fafc;letter-spacing:-0.03em;">'
        'Blend<span style="color:#6366f1;">WX</span></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bwx-sidebar-label">Location</div>', unsafe_allow_html=True)

    loc_mode = st.radio("Input method", ["City search", "Lat / Lon"], horizontal=True, label_visibility="collapsed")

    lat = lon = None
    location_label = None
    chosen_geo: dict = {}

    if loc_mode == "City search":
        q = st.text_input("City", placeholder="e.g., Toronto", label_visibility="collapsed")
        results = []
        if q.strip():
            try:
                results = geocode_city(q, count=10)
            except Exception as e:
                st.error(f"Geocoding failed: {e}")

        if results:
            options = []
            for r in results:
                name = r.get("name")
                admin1 = r.get("admin1")
                country = r.get("country")
                tz = r.get("timezone")
                label = f"{name}, {admin1 or ''} {country or ''}".replace("  ", " ").strip()
                if tz:
                    label += f" ({tz})"
                options.append(label)

            sel = st.selectbox("Match", options, index=0, label_visibility="collapsed")
            chosen_geo = results[options.index(sel)]
            lat = float(chosen_geo["latitude"])
            lon = float(chosen_geo["longitude"])
            location_label = sel
        elif q.strip():
            st.info(f"No results for '{q}'.")
        else:
            st.caption("Type a city name to search.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Lat", value=43.6532, format="%.4f")
        with c2:
            lon = st.number_input("Lon", value=-79.3832, format="%.4f")
        location_label = f"Lat {lat:.4f}, Lon {lon:.4f}"

    st.divider()
    st.markdown('<div class="bwx-sidebar-label">Forecast horizon</div>', unsafe_allow_html=True)
    horizon_labels = [f"{d}D" for d in HORIZON_OPTIONS]
    horizon_idx = st.radio(
        "Horizon",
        options=range(len(HORIZON_OPTIONS)),
        format_func=lambda i: horizon_labels[i],
        index=HORIZON_OPTIONS.index(DEFAULT_HORIZON),
        horizontal=True,
        label_visibility="collapsed",
    )
    horizon = HORIZON_OPTIONS[horizon_idx]

    st.markdown('<div class="bwx-sidebar-label">Display</div>', unsafe_allow_html=True)
    display_mode = st.radio(
        "Display mode",
        ["Models only", "Blend only", "Models + Blend"],
        index=2,
        label_visibility="collapsed",
        help="Blend is the mean across selected models at each timestamp.",
    )

    st.markdown('<div class="bwx-sidebar-label">Precip bar interval</div>', unsafe_allow_html=True)
    bar_interval = st.selectbox("Bar interval", PRECIP_BAR_INTERVALS, index=PRECIP_BAR_INTERVALS.index(DEFAULT_BAR_INTERVAL), label_visibility="collapsed")

    st.markdown('<div class="bwx-sidebar-label">Units</div>', unsafe_allow_html=True)
    units = st.radio("Units", ["Metric", "Imperial"], horizontal=True, label_visibility="collapsed")
    imperial = units == "Imperial"

    st.divider()
    st.markdown('<div class="bwx-sidebar-label">Models</div>', unsafe_allow_html=True)
    models = []
    for model_name, color in [(m, MODEL_COLORS[m]) for m in ["ECMWF IFS", "GFS Seamless", "GEM Seamless"]]:
        dot = (
            f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
            f'background:{color};margin-right:6px;vertical-align:middle;"></span>'
        )
        on = st.toggle(
            f"{model_name}",
            value=model_name in DEFAULT_MODELS,
            key=f"toggle_{model_name}",
        )
        if on:
            models.append(model_name)

    st.divider()
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Guardrails
if lat is None or lon is None:
    st.stop()
if not models:
    st.warning("Select at least one model.")
    st.stop()

# -----------------------------
# Fetch
# -----------------------------
with st.spinner("Fetching forecasts…"):
    frames = []
    tz_labels = set()
    resolved = []

    for m in models:
        try:
            df_m = fetch_model_with_candidates(m, lat, lon, forecast_days=horizon)
            tz_labels.add(df_m.attrs.get("timezone", "UTC"))
            resolved.append({"Model": m, "Resolved model slug": df_m.attrs.get("model_slug", "")})
            frames.append(df_m)
        except Exception as e:
            st.error(f"Failed to fetch {m}: {e}")

df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if df_all.empty:
    st.error("No data returned for the selected inputs.")
    st.stop()

# Add blend depending on mode
if display_mode in ["Blend only", "Models + Blend"]:
    df_all = add_blend(df_all, label="Blend")

# Always have a blended df for cards/headline (even in Models only mode)
df_blend_all = df_all if "Blend" in df_all["model"].unique() else add_blend(df_all)

# Time bounds
min_ts = df_all["timestamp"].min()
max_ts = df_all["timestamp"].max()

# Build breadcrumb parts from geocode result if available
_city    = chosen_geo.get("name", "")
_region  = chosen_geo.get("admin1", "")
_country = chosen_geo.get("country", "")
if _city and _region and _country:
    _breadcrumb = f'<span style="font-size:0.78rem;color:rgba(248,250,252,0.35);letter-spacing:0.03em;">{_country} › {_region} › {_city}</span>'
    _city_display = _city
elif _city and _country:
    _breadcrumb = f'<span style="font-size:0.78rem;color:rgba(248,250,252,0.35);letter-spacing:0.03em;">{_country} › {_city}</span>'
    _city_display = _city
else:
    _breadcrumb = ""
    _city_display = location_label or ""

_tz_str  = ', '.join(sorted(tz_labels))
_win_str = f"{min_ts.strftime('%b %d')} – {max_ts.strftime('%b %d, %Y')}"
_lat_lon = f"{lat:.3f}, {lon:.3f}" if lat is not None else ""

st.markdown(
    f'<div style="margin-bottom:4px;">{_breadcrumb}</div>'
    f'<div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:8px;">'
    f'  <span style="font-size:1.7rem;font-weight:700;color:#f8fafc;letter-spacing:-0.03em;">{_city_display}</span>'
    f'  <span style="font-size:0.85rem;color:#6366f1;font-family:monospace;">{_lat_lon}</span>'
    f'</div>'
    f'<div style="display:flex;flex-wrap:wrap;margin-bottom:12px;">'
    f'  <span class="bwx-pill">🕐 {_tz_str}</span>'
    f'  <span class="bwx-pill">📅 {_win_str}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

headline = compute_headline(df_blend_all)
if headline:
    st.markdown(f'<div class="bwx-headline">{headline}</div>', unsafe_allow_html=True)

render_daily_cards(df_blend_all, imperial)

# Time window slider
default_start = min_ts
default_end = min(max_ts, min_ts + pd.Timedelta(days=horizon))

start_dt, end_dt = st.slider(
    "Time window",
    min_value=min_ts.to_pydatetime(),
    max_value=max_ts.to_pydatetime(),
    value=(default_start.to_pydatetime(), default_end.to_pydatetime()),
    format="MMM D, YYYY",
)

start_ts = pd.Timestamp(start_dt)
end_ts = pd.Timestamp(end_dt)
if start_ts.tzinfo is None:
    start_ts = start_ts.tz_localize(min_ts.tz)
if end_ts.tzinfo is None:
    end_ts = end_ts.tz_localize(min_ts.tz)

df = df_all[(df_all["timestamp"] >= start_ts) & (df_all["timestamp"] <= end_ts)].copy()

# Apply unit conversion
df = convert_units(df, imperial)

# Dynamic unit labels
t_unit = "°F" if imperial else "°C"
w_unit = "mph" if imperial else "km/h"
p_unit = "in"  if imperial else "mm"

st.divider()

_chart_panel_header("Temperature", df, display_mode)
plot_timeseries(df, "temp_c", "Air temperature", f"Temperature ({t_unit})", t_unit, start_ts, end_ts, display_mode)

_chart_panel_header("Feels-like temperature", df, display_mode)
plot_timeseries(df, "feels_like_c", "Feels-like temperature", f"Feels-like ({t_unit})", t_unit, start_ts, end_ts, display_mode)

_chart_panel_header("Relative humidity", df, display_mode)
plot_timeseries(df, "rh_pct", "Relative humidity", "Humidity (%)", "%", start_ts, end_ts, display_mode, y_range=[0, 100])

_chart_panel_header("Cloud cover", df, display_mode)
plot_timeseries(df, "cloud_pct", "Cloud cover", "Cloud cover (%)", "%", start_ts, end_ts, display_mode, y_range=[0, 100], smooth_window=3)

_chart_panel_header("Sunshine hours", df, display_mode)
plot_daily_sunshine(compute_daily_sunshine(df), display_mode, start_ts, end_ts)

_chart_panel_header(f"Precipitation — {bar_interval} totals", df, display_mode)
plot_precip_bars(df, f"{bar_interval} totals", start_ts, end_ts, display_mode, bar_interval, p_unit=p_unit)

_chart_panel_header("Cumulative precipitation", df, display_mode)
plot_precip_cumulative(df, "Cumulative total", start_ts, end_ts, display_mode, p_unit=p_unit)

_chart_panel_header("Probability of precipitation", df, display_mode)
plot_timeseries(df, "precip_prob_pct", "Probability of precipitation", "Probability (%)", "%", start_ts, end_ts, display_mode, y_range=[0, 100])

_chart_panel_header("Wind Speed", df, display_mode)
plot_timeseries(df, "wind_kmh", "Wind speed", f"Wind speed ({w_unit})", w_unit, start_ts, end_ts, display_mode, to_zero=True, smooth_window=3)

_chart_panel_header("Wind Gusts", df, display_mode)
plot_timeseries(df, "wind_gust_kmh", "Wind gusts", f"Wind gusts ({w_unit})", w_unit, start_ts, end_ts, display_mode, to_zero=True, smooth_window=3)

_chart_panel_header("Surface pressure", df, display_mode)
plot_timeseries(df, "pressure_hpa", "Surface pressure", "hPa", "hPa", start_ts, end_ts, display_mode)


st.divider()
st.subheader("Export")
cols = [
    "timestamp",
    "model",
    "temp_c",
    "feels_like_c",
    "rh_pct",
    "cloud_pct",
    "precip_mm",
    "precip_prob_pct",
    "wind_kmh",
    "wind_gust_kmh",
    "pressure_hpa",
]
export_df = df[[c for c in cols if c in df.columns]].copy()

export_df["timestamp"] = pd.to_datetime(export_df["timestamp"]).dt.strftime("%a %b %d, %Y %I:%M %p %Z")

for c in export_df.columns:
    if c not in ("timestamp", "model"):
        export_df[c] = pd.to_numeric(export_df[c], errors="coerce").round(1)

csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="blendwx_hourly.csv", mime="text/csv")

st.markdown(
    '<div class="bwx-footer">'
    'Forecast data via <a href="https://open-meteo.com" target="_blank">Open-Meteo</a> &nbsp;·&nbsp;'
    'Models: ECMWF IFS &nbsp;·&nbsp; NOAA GFS &nbsp;·&nbsp; Environment Canada GEM'
    '</div>',
    unsafe_allow_html=True,
)
