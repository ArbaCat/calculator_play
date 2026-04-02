"""
DAITABLE ENERGY OPTIMIZATION DASHBOARD
======================================
Full-featured BESS + Solar PV profit calculator and operational simulator.
Modules:
  1. Design & Utilities      - Brand palette, theme, KPI card helpers
  2. Real Data Ingestion     - OKTE CSV price reader
  3. Energy Forecasting      - 12-month history generation + Prophet/sklearn forecast
  4. Solar Power Model       - Simplified bell-curve PV generation
  5. LP Optimization         - PuLP-based 24-hour dispatch plan
  6. Main UI & Visualizations - Streamlit layout, charts, KPIs
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# LP solver
import pulp

# Forecasting – try Prophet first, fall back to sklearn ridge
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1 – DESIGN & UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

# ── Brand palette ──────────────────────────────────────────────────────────
C_GREEN  = "#26B24B"
C_WHITE  = "#FFFFFF"
C_BLACK  = "#000000"
C_GREY1  = "#1E1E1E"   # card backgrounds
C_GREY2  = "#1E1E26"   # borders / detail cards
C_AMBER  = "#F5A623"   # accent for warnings / discharge
C_BLUE   = "#4A90D9"   # accent for solar / SoC

# ── Streamlit page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="DAITABLE Energy Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS injection ────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Full page dark background */
  .stApp {{ background-color: {C_BLACK}; color: {C_WHITE}; }}
  section[data-testid="stSidebar"] {{
    background-color: {C_GREY1};
    border-right: 1px solid {C_GREY2};
  }}
  /* Input labels */
  label, .stSlider label, .stSelectbox label {{ color: {C_WHITE} !important; }}
  /* Metric cards default override */
  div[data-testid="metric-container"] {{
    background-color: {C_GREY1};
    border: 1px solid {C_GREY2};
    border-radius: 8px;
    padding: 12px;
  }}
  div[data-testid="metric-container"] label {{ color: #AAAAAA !important; font-size: 0.8rem; }}
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
    color: {C_WHITE} !important; font-size: 1.6rem; font-weight: 700;
  }}
  /* Section headers */
  .section-header {{
    color: {C_GREEN};
    font-size: 1.1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid {C_GREEN};
    padding-bottom: 6px;
    margin-bottom: 16px;
  }}
  /* KPI card (custom HTML) */
  .kpi-card {{
    background-color: {C_GREY1};
    border: 1px solid {C_GREY2};
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
  }}
  .kpi-card.green {{
    background-color: {C_GREEN};
    border-color: {C_GREEN};
  }}
  .kpi-card .kpi-label {{
    font-size: 0.75rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #BBBBBB;
    margin-bottom: 6px;
  }}
  .kpi-card.green .kpi-label {{ color: #e0ffe8; }}
  .kpi-card .kpi-value {{
    font-size: 2rem;
    font-weight: 800;
    color: {C_WHITE};
  }}
  .kpi-card .kpi-sub {{
    font-size: 0.8rem;
    color: #AAAAAA;
    margin-top: 4px;
  }}
  .kpi-card.green .kpi-sub {{ color: #e0ffe8; }}
  /* Footer */
  .footer {{
    color: #555555;
    font-size: 0.7rem;
    text-align: right;
    padding-top: 24px;
    border-top: 1px solid {C_GREY2};
  }}
</style>
""", unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", green: bool = False) -> str:
    """Return an HTML KPI card string."""
    cls = "kpi-card green" if green else "kpi-card"
    return f"""
    <div class="{cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """


def plotly_base_layout(title: str = "") -> dict:
    """Return a base Plotly layout dict matching the DAITABLE dark theme."""
    return dict(
        title=dict(text=title, font=dict(color=C_WHITE, size=14)),
        paper_bgcolor=C_GREY1,
        plot_bgcolor=C_GREY2,
        font=dict(color=C_WHITE, family="sans-serif"),
        xaxis=dict(
            gridcolor="#2A2A38",
            linecolor="#444",
            tickfont=dict(color="#AAAAAA"),
        ),
        yaxis=dict(
            gridcolor="#2A2A38",
            linecolor="#444",
            tickfont=dict(color="#AAAAAA"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="#444",
            font=dict(color=C_WHITE),
        ),
        margin=dict(l=50, r=20, t=50, b=40),
    )


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2 – REAL DATA INGESTION (OKTE price CSV)
# ═════════════════════════════════════════════════════════════════════════════

def load_okte_prices(csv_path: str) -> pd.Series:
    """
    Load hourly day-ahead electricity prices from an OKTE-formatted CSV.

    Expected columns: Date, Hour (0-23), Price_EUR_MWh
    Returns a pd.Series of 24 float values (EUR/MWh), indexed 0..23.
    """
    if not os.path.exists(csv_path):
        st.warning(f"Price file not found: {csv_path}. Using synthetic prices.")
        return _synthetic_prices()

    df = pd.read_csv(csv_path)
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    if "Hour" not in df.columns or "Price_EUR_MWh" not in df.columns:
        st.warning("CSV format mismatch – expected 'Hour' and 'Price_EUR_MWh' columns.")
        return _synthetic_prices()

    prices = df.sort_values("Hour")["Price_EUR_MWh"].values[:24].astype(float)
    return pd.Series(prices, index=range(24), name="Price_EUR_MWh")


def _synthetic_prices() -> pd.Series:
    """Fallback: generate a realistic Slovak day-ahead price curve (EUR/MWh)."""
    base = np.array([
        62, 55, 51, 49, 49, 53, 79, 112, 126, 118, 105, 98,
        92, 89, 94, 102, 116, 138, 152, 145, 128, 109, 88, 71
    ], dtype=float)
    rng = np.random.default_rng(42)
    return pd.Series(base + rng.normal(0, 2, 24), index=range(24), name="Price_EUR_MWh")


# ── Default path for real historical price file ────────────────────────────
DEFAULT_PRICE_HISTORY_CSV = "okte_prices_history.csv"


@st.cache_data(show_spinner=False)
def load_real_price_history(csv_path: str) -> pd.Series:
    """
    Load 15-minute resolution real electricity prices, strip the '€' symbol,
    convert to float and resample to hourly mean.

    Input CSV columns: price (e.g. '€120.98'), time (datetime string)

    Returns hourly pd.Series (EUR/MWh) with DatetimeIndex, name='Price_EUR_MWh'.
    """
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df["Price_EUR_MWh"] = (
        df["price"].astype(str)
        .str.replace("€", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    df = df.set_index("time").sort_index()
    hourly = df["Price_EUR_MWh"].resample("1h").mean()
    hourly.name = "Price_EUR_MWh"
    return hourly


@st.cache_data(show_spinner=False)
def forecast_prices_24h(price_history_json: str, ref_dt_str: str) -> pd.Series:
    """
    Return 24 hourly prices (EUR/MWh) for the requested date.

    Strategy:
      - If date exists in real history (≥20 h present): return actual values.
      - Otherwise: profile-based forecast.

    Profile model (much better than Prophet for spiky spot prices):
      1. Compute mean price for each (day-of-week × hour) from the last 4 weeks.
         This captures the real intra-day + weekend/weekday pattern.
      2. Apply a "level correction" = ratio of last-7d mean to overall mean,
         so recent price regime is respected.
      3. Light Gaussian smoothing to remove remaining hour-to-hour jitter.

    Arguments are JSON strings for st.cache_data hashability.
    Returns pd.Series of 24 floats indexed 0..23.
    """
    from scipy.ndimage import gaussian_filter1d  # for smoothing

    history = pd.read_json(io.StringIO(price_history_json), typ="series")
    history.index = pd.to_datetime(history.index)
    history = history.sort_index()
    ref_dt  = pd.to_datetime(ref_dt_str)

    # ── 1. Use real data if available for this exact date ─────────────────
    day_mask = history.index.date == ref_dt.date()
    if day_mask.sum() >= 20:
        day_prices = history[day_mask].reindex(
            pd.date_range(ref_dt, periods=24, freq="1h")
        ).ffill().bfill()
        return pd.Series(day_prices.values, index=range(24), name="Price_EUR_MWh")

    # ── 2. Profile-based forecast ─────────────────────────────────────────
    df = history.to_frame("price")
    df["hour"] = df.index.hour
    df["dow"]  = df.index.dayofweek

    # Use last 4 weeks of history to build the profile (more recent = more relevant)
    recent_4w = df.tail(4 * 7 * 24)
    # (dow, hour) → mean price
    profile = recent_4w.groupby(["dow", "hour"])["price"].mean()
    # Fallback: full-history profile for any missing (dow, hour) cells
    full_profile = df.groupby(["dow", "hour"])["price"].mean()

    # Level correction: recent 7d average vs overall average
    level_recent = df.tail(7 * 24)["price"].mean()
    level_all    = df["price"].mean()
    level_ratio  = level_recent / level_all if level_all > 0 else 1.0
    # Cap the correction to ±40% to avoid overreaction to outlier weeks
    level_ratio  = float(np.clip(level_ratio, 0.60, 1.40))

    # Build 24 forecast values
    vals = []
    for h in range(24):
        dow_fc = ref_dt.weekday()
        # Prefer recent profile; fall back to full profile; final fallback = overall mean
        base = profile.get((dow_fc, h),
               full_profile.get((dow_fc, h), level_all))
        vals.append(base * level_ratio)

    vals = np.array(vals, dtype=float)

    # Light Gaussian smoothing (σ=0.8 → ~1-2h blur, removes hour-to-hour jitter)
    vals = gaussian_filter1d(vals, sigma=0.8)
    vals = np.clip(vals, 0, 500)

    return pd.Series(vals, index=range(24), name="Price_EUR_MWh")


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2b – REAL FACILITY DATA LOADER & PERIOD PRICE GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

# Default path – can be overridden from sidebar
DEFAULT_REAL_CSV = "facility_load.csv"

# Friendly display names for the measurements in the CSV
MEAS_LABELS = {
    "Hospital_Total": "Hospital Total Load",
    # legacy / fallback names kept for compatibility
    "Generator":          "Diesel Generator",
    "Distribution_Main":  "Thermal / Heating",
    "HVAC_Main":          "HVAC / Ventilation",
}
MEAS_COLORS = {
    "Hospital_Total":     C_BLUE,
    "Generator":          C_AMBER,
    "Distribution_Main":  "#E05C5C",
    "HVAC_Main":          C_BLUE,
}


@st.cache_data(show_spinner=False)
def load_real_facility_data(csv_path: str) -> pd.DataFrame:
    """
    Load sub-minute facility power data, pivot to wide format and resample
    to hourly means.

    Input CSV columns: datetime, measurement, Active_Power_Total (Watts)

    Returns hourly DataFrame (DatetimeIndex, UTC-naive) with:
      - one column per measurement in kW
      - 'Total_kW' = sum of all measurements
    """
    df = pd.read_csv(csv_path)
    # Handle both UTC-aware ("+00:00" suffix) and naive local timestamps
    _dt = pd.to_datetime(df["datetime"], errors="coerce")
    if _dt.dt.tz is not None:
        _dt = _dt.dt.tz_convert(None)   # tz-aware → UTC-naive
    else:
        _dt = _dt.dt.tz_localize(None)  # already naive, just ensure no tz
    df["datetime"] = _dt

    # Pivot: one row per timestamp, one column per measurement
    piv = df.pivot_table(
        index="datetime", columns="measurement",
        values="Active_Power_Total", aggfunc="mean",
    )
    piv.columns.name = None
    piv = piv / 1000.0  # W → kW

    # Resample to hourly mean, fill tiny gaps (≤2 h) by forward-fill
    piv = piv.resample("1h").mean().ffill(limit=2)
    piv["Total_kW"] = piv.sum(axis=1)
    return piv


def generate_price_series_range(
    date_index: pd.DatetimeIndex,
    price_history_csv: str = DEFAULT_PRICE_HISTORY_CSV,
    seed: int = 7,
    winter_premium: float = 1.15,
) -> pd.Series:
    """
    Return hourly electricity prices (EUR/MWh) aligned to date_index.

    Priority:
      1. Real prices from price_history_csv where available (exact hourly match).
      2. Synthetic curve for any hours not covered by real data.
    """
    # ── Try to load real price history ────────────────────────────────────
    real_prices = None
    if os.path.exists(price_history_csv):
        try:
            real_prices = load_real_price_history(price_history_csv)
        except Exception:
            pass

    if real_prices is not None:
        # Reindex to requested date_index, forward-fill gaps up to 3h
        aligned = real_prices.reindex(date_index).ffill(limit=3).bfill(limit=3)
        missing_mask = aligned.isna()
    else:
        aligned      = pd.Series(np.nan, index=date_index)
        missing_mask = pd.Series(True, index=date_index)

    # ── Synthetic fallback for any uncovered hours ─────────────────────────
    if missing_mask.any():
        rng   = np.random.default_rng(seed)
        hours = date_index.hour.values
        dow   = date_index.dayofweek.values
        month = date_index.month.values

        base_curve = np.array([
            62, 55, 51, 49, 49, 53, 79, 112, 126, 118, 105, 98,
            92, 89, 94, 102, 116, 138, 152, 145, 128, 109, 88, 71
        ], dtype=float)

        synth = base_curve[hours]
        synth *= np.where(dow < 5, 1.0, 0.75)
        seasonal = np.where((month <= 2) | (month >= 11), winter_premium, 1.0)
        synth   *= seasonal
        day_dates   = date_index.normalize()
        unique_days = sorted(set(day_dates))
        day_mults   = {d: rng.uniform(0.85, 1.15) for d in unique_days}
        synth *= np.array([day_mults[d] for d in day_dates])
        synth += rng.normal(0, 3, len(date_index))
        synth  = np.clip(synth, 20, 350)

        aligned[missing_mask] = synth[missing_mask.values]

    aligned.name = "Price_EUR_MWh"
    return aligned


@st.cache_data(show_spinner=False)
def run_period_simulation(
    # Serialisable primitives only (for st.cache_data hashing)
    load_json: str,          # hourly Total_kW as JSON string
    price_json: str,         # hourly prices as JSON string
    capacity_kwh: float,
    max_power_kw: float,
    charge_eff: float,
    discharge_eff: float,
    soc_min_pct: float,
    soc_max_pct: float,
    pv_peak_kw: float,
    pv_efficiency: float,
    latitude: float,
) -> pd.DataFrame:
    """
    Run the LP dispatch optimiser day-by-day over the full historical period.

    Returns a DataFrame with one row per day:
      Date | Baseline_Cost_EUR | Optimised_Cost_EUR | Savings_EUR |
      Total_Load_kWh | Total_Solar_kWh | LP_Status
    """
    hourly_load   = pd.read_json(io.StringIO(load_json),  typ="series")
    hourly_prices = pd.read_json(io.StringIO(price_json), typ="series")

    # Ensure DatetimeIndex
    hourly_load.index   = pd.to_datetime(hourly_load.index)
    hourly_prices.index = pd.to_datetime(hourly_prices.index)

    unique_days = sorted(set(hourly_load.index.normalize()))
    results = []

    for day in unique_days:
        day_load   = hourly_load[hourly_load.index.date   == day.date()]
        day_prices = hourly_prices[hourly_prices.index.date == day.date()]

        if len(day_load) < 24 or len(day_prices) < 24:
            continue

        day_load   = pd.Series(day_load.values[:24],   index=range(24))
        day_prices = pd.Series(day_prices.values[:24], index=range(24))

        solar = solar_generation_forecast(
            peak_kw=pv_peak_kw,
            efficiency=pv_efficiency,
            latitude_deg=latitude,
            month=day.month,
            day_noise_seed=day.day + day.month * 31,
        )

        res = optimize_bess_dispatch(
            load_kw=day_load, solar_kw=solar, prices_eur_mwh=day_prices,
            capacity_kwh=capacity_kwh, max_power_kw=max_power_kw,
            charge_eff=charge_eff, discharge_eff=discharge_eff,
            soc_min_pct=soc_min_pct, soc_max_pct=soc_max_pct,
            initial_soc_pct=0.50,
        )

        results.append({
            "Date":               day.date(),
            "GridOnly_Cost_EUR":  res["cost_grid_only"],  # no solar, no BESS
            "Baseline_Cost_EUR":  res["cost_base"],       # solar only, no BESS
            "Optimised_Cost_EUR": res["cost_optimised"],  # solar + BESS
            "Savings_EUR":        res["savings"],          # vs grid-only
            "Total_Load_kWh":     float(day_load.sum()),
            "Total_Solar_kWh":    float(solar.sum()),
            "LP_Status":          res["status"],
        })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3 – ENERGY FORECASTING & FACILITY CONSUMPTION
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_historical_consumption(seed: int = 0) -> pd.DataFrame:
    """
    Generate 12 months of realistic hourly facility consumption data.

    Features modelled:
      - Base load (always-on HVAC, lighting, servers)
      - Heavy-machinery spikes (deterministic shift windows + random triggers)
      - Day-of-week variation (weekdays > weekends)
      - Seasonal variation (higher in winter/summer, lower in spring/autumn)
      - Gaussian noise
    Returns DataFrame with columns: [ds, y] (Prophet-compatible naming).
    """
    rng = np.random.default_rng(seed)
    start = datetime(2025, 4, 1)           # 12 months back from ~2026-03
    periods = 365 * 24
    timestamps = [start + timedelta(hours=i) for i in range(periods)]
    df = pd.DataFrame({"ds": timestamps})

    hours   = df["ds"].dt.hour.values
    dow     = df["ds"].dt.dayofweek.values   # 0=Mon … 6=Sun
    month   = df["ds"].dt.month.values

    # ── Base load (kW) ──────────────────────────────────────────────────
    base = 120.0

    # ── Day-of-week factor ──────────────────────────────────────────────
    dow_factor = np.where(dow < 5, 1.0, 0.55)   # weekday vs weekend

    # ── Seasonal factor (cosine, peaks Jan & Jul) ───────────────────────
    seasonal = 1.0 + 0.18 * np.cos((month - 1) / 6 * np.pi)

    # ── Diurnal profile (double-peak: morning + afternoon ramp) ─────────
    diurnal = (
        0.30
        + 0.45 * np.exp(-((hours - 9) ** 2) / 12)
        + 0.35 * np.exp(-((hours - 15) ** 2) / 14)
        - 0.15 * np.exp(-((hours - 3)  ** 2) / 8)
    )
    diurnal = np.clip(diurnal, 0.2, 1.0)

    # ── Heavy-machinery spikes (industrial shifts 06-14 & 14-22 weekdays) ──
    shift_morning   = ((hours >= 6)  & (hours < 14) & (dow < 5)).astype(float)
    shift_afternoon = ((hours >= 14) & (hours < 22) & (dow < 5)).astype(float)
    machinery = (shift_morning * 80 + shift_afternoon * 60)
    # Random short bursts (compressors, etc.)
    bursts = rng.poisson(0.04, periods) * rng.uniform(20, 60, periods)
    bursts *= (dow < 5)

    # ── Noise ──────────────────────────────────────────────────────────
    noise = rng.normal(0, 5, periods)

    consumption = base * dow_factor * seasonal * diurnal + machinery + bursts + noise
    consumption = np.clip(consumption, 30, 500)

    df["y"] = consumption
    return df


@st.cache_data(show_spinner=False)
def train_forecast_model(history_df: pd.DataFrame):
    """
    Train a forecasting model on 12 months of historical hourly data.

    If Prophet is available, fit a Prophet model with daily + weekly seasonality.
    Otherwise, fall back to a Ridge regression on engineered time features.

    Returns (model, scaler_or_None, model_type_str).
    """
    if PROPHET_AVAILABLE:
        m = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.1,
            interval_width=0.80,
        )
        m.fit(history_df[["ds", "y"]])
        return m, None, "Prophet"

    # ── Sklearn Ridge fallback ─────────────────────────────────────────
    def make_features(ds_series: pd.Series) -> np.ndarray:
        h  = ds_series.dt.hour.values
        dw = ds_series.dt.dayofweek.values
        mo = ds_series.dt.month.values
        feat = np.stack([
            np.sin(2 * np.pi * h  / 24),
            np.cos(2 * np.pi * h  / 24),
            np.sin(2 * np.pi * dw / 7),
            np.cos(2 * np.pi * dw / 7),
            np.sin(2 * np.pi * mo / 12),
            np.cos(2 * np.pi * mo / 12),
            (dw < 5).astype(float),
        ], axis=1)
        return feat

    X = make_features(history_df["ds"])
    y = history_df["y"].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    model  = Ridge(alpha=10.0)
    model.fit(X_sc, y)
    # Store make_features inside scaler for convenience
    scaler._make_features = make_features
    return model, scaler, "Ridge"


def forecast_next_24h(model, scaler, model_type: str,
                      ref_date: datetime) -> pd.Series:
    """
    Generate a 24-hour load forecast starting from ref_date hour 0.

    Returns pd.Series of 24 floats (kW), indexed 0..23.
    """
    future_hours = [ref_date + timedelta(hours=h) for h in range(24)]

    if model_type == "Prophet":
        future_df = pd.DataFrame({"ds": future_hours})
        forecast  = model.predict(future_df)
        vals = forecast["yhat"].values
    else:
        X_fut = scaler._make_features(pd.Series(future_hours))
        X_sc  = scaler.transform(X_fut)
        vals  = model.predict(X_sc)

    vals = np.clip(vals, 20, 600)
    return pd.Series(vals, index=range(24), name="LoadForecast_kW")


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4 – SIMPLE SOLAR POWER MODEL
# ═════════════════════════════════════════════════════════════════════════════

def solar_generation_forecast(
    peak_kw: float,
    efficiency: float = 0.18,
    latitude_deg: float = 48.5,   # Slovakia default
    month: int = 6,
    day_noise_seed: int = 1,
) -> pd.Series:
    """
    Simplified hourly PV generation for a single day using a Gaussian bell curve.

    - Solar noon shift based on latitude & month (approximate).
    - Bell-curve width modulated by declination (summer wider, winter narrower).
    - Efficiency applied on top of peak_kw.

    Returns pd.Series of 24 floats (kW), indexed 0..23.
    """
    # Approximate solar declination (degrees)
    day_of_year = {1:15, 2:46, 3:75, 4:105, 5:135, 6:162,
                   7:198, 8:228, 9:258, 10:288, 11:318, 12:344}.get(month, 172)
    declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))

    # Sunrise/sunset hours (approx)
    cos_ha = -np.tan(np.radians(latitude_deg)) * np.tan(np.radians(declination))
    cos_ha = np.clip(cos_ha, -1, 1)
    half_day = np.degrees(np.arccos(cos_ha)) / 15   # hours from noon
    sunrise = max(4.0, 12.0 - half_day)
    sunset  = min(22.0, 12.0 + half_day)
    width   = (sunset - sunrise) / 4.5   # sigma: ~5-6 peak-sun-hours for Slovakia (was /2.5 → 2× overestimate)

    hours  = np.arange(24)
    noon   = (sunrise + sunset) / 2
    bell   = np.exp(-0.5 * ((hours - noon) / width) ** 2)
    bell   = np.where((hours >= sunrise) & (hours <= sunset), bell, 0.0)

    # Apply peak & efficiency, then scale to match real monthly yield for Slovakia.
    # Raw bell integral overestimates by ~2× because it ignores clouds, panel tilt,
    # inverter losses, and low sun angle.  We use PVGIS-derived monthly kWh/kWp
    # values for central Slovakia (south-facing 30° tilt) as ground truth.
    _SK_YIELD = {1:1.0, 2:1.8, 3:3.0, 4:3.8, 5:4.2, 6:4.5,
                 7:4.5, 8:4.0, 9:3.2, 10:2.0, 11:1.0, 12:0.7}
    raw_integral = bell.sum()  # kWh/kWp from raw gaussian
    target_yield = _SK_YIELD.get(month, 3.0)
    scale = target_yield / raw_integral if raw_integral > 0.01 else 0.0

    gen = peak_kw * efficiency / 0.18 * bell * scale
    gen = np.clip(gen, 0, peak_kw)

    # Mild day-to-day cloud noise centered at 1.0 (±15%, no bias)
    rng = np.random.default_rng(day_noise_seed)
    cloud = 1.0 + (rng.beta(5, 5, 24) - 0.5) * 0.3
    gen   = gen * cloud

    return pd.Series(gen, index=range(24), name="SolarForecast_kW")


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5 – LP OPTIMIZATION (PuLP)
# ═════════════════════════════════════════════════════════════════════════════

def optimize_bess_dispatch(
    load_kw:        pd.Series,
    solar_kw:       pd.Series,
    prices_eur_mwh: pd.Series,
    capacity_kwh:   float = 500.0,
    max_power_kw:   float = 125.0,
    charge_eff:     float = 0.9487,   # sqrt(0.90) → 90 % round-trip default, matches UI
    discharge_eff:  float = 0.9487,
    soc_min_pct:    float = 0.10,
    soc_max_pct:    float = 0.95,
    initial_soc_pct: float = 0.50,
) -> dict:
    """
    Solve a 24-hour BESS dispatch optimisation using Linear Programming (PuLP).

    Objective : Minimise total grid electricity cost.
    Variables  : GridImport[t], BattCharge[t], BattDischarge[t], SoC[t]
    Returns    : dict with keys 'status', 'schedule' (DataFrame), 'cost_optimised',
                 'cost_base', 'savings'
    """
    T = 24
    hours = list(range(T))

    # Convert prices from EUR/MWh to EUR/kWh
    prices_kwh = prices_eur_mwh.values / 1000.0

    soc_min = soc_min_pct * capacity_kwh
    soc_max = soc_max_pct * capacity_kwh
    soc_init = initial_soc_pct * capacity_kwh

    # ── LP Problem ─────────────────────────────────────────────────────
    prob = pulp.LpProblem("BESS_Dispatch", pulp.LpMinimize)

    # Decision variables
    grid    = [pulp.LpVariable(f"grid_{t}",    lowBound=0) for t in hours]
    charge  = [pulp.LpVariable(f"charge_{t}",  lowBound=0, upBound=max_power_kw) for t in hours]
    disch   = [pulp.LpVariable(f"disch_{t}",   lowBound=0, upBound=max_power_kw) for t in hours]
    soc     = [pulp.LpVariable(f"soc_{t}",     lowBound=soc_min, upBound=soc_max) for t in hours]
    # Curtailment: excess solar that cannot be used or stored (no export assumed)
    curtail = [pulp.LpVariable(f"curtail_{t}", lowBound=0) for t in hours]

    # ── Objective ──────────────────────────────────────────────────────
    prob += pulp.lpSum(prices_kwh[t] * grid[t] for t in hours), "TotalCost"

    # ── Constraints ────────────────────────────────────────────────────
    for t in hours:
        solar_t = solar_kw.iloc[t]
        load_t  = load_kw.iloc[t]

        # Power balance:
        #   Load = Solar_used + Grid + Discharge - Charge
        # where Solar_used = solar_t - curtail_t
        # Rearranged: Grid = Load - solar_t + curtail_t + Charge - Discharge
        prob += (
            grid[t] == load_t - solar_t + curtail[t] + charge[t] - disch[t],
            f"PowerBalance_{t}",
        )
        # Grid import cannot be negative (no export)
        prob += grid[t] >= 0, f"NoExport_{t}"
        # Curtailment cannot exceed available solar
        prob += curtail[t] <= solar_t, f"CurtailLimit_{t}"

        # SoC dynamics
        if t == 0:
            prob += (
                soc[t] == soc_init + charge[t] * charge_eff - disch[t] / discharge_eff,
                "SoC_0",
            )
        else:
            prob += (
                soc[t] == soc[t-1] + charge[t] * charge_eff - disch[t] / discharge_eff,
                f"SoC_{t}",
            )

        # Cannot charge AND discharge simultaneously
        bin_cd = pulp.LpVariable(f"bin_{t}", cat="Binary")
        prob += charge[t] <= max_power_kw * bin_cd,       f"ChargeBin_{t}"
        prob += disch[t]  <= max_power_kw * (1 - bin_cd), f"DischBin_{t}"

    # ── Solve ──────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    cost_grid_only = float(np.sum(load_kw.values * prices_kwh))  # no solar, no BESS
    if status not in ("Optimal", "Feasible"):
        # Return unoptimised baseline
        base_cost = float(np.sum(
            np.maximum(load_kw.values - solar_kw.values, 0) * prices_kwh
        ))
        return {
            "status": status,
            "schedule": None,
            "cost_optimised": base_cost,
            "cost_base": base_cost,
            "cost_grid_only": cost_grid_only,
            "savings": 0.0,
        }

    # ── Extract results ────────────────────────────────────────────────
    schedule = pd.DataFrame({
        "Hour":        hours,
        "Load_kW":     load_kw.values,
        "Solar_kW":    solar_kw.values,
        "Charge_kW":   [pulp.value(charge[t])  or 0 for t in hours],
        "Disch_kW":    [pulp.value(disch[t])   or 0 for t in hours],
        "Grid_kW":     [pulp.value(grid[t])    or 0 for t in hours],
        "Curtail_kW":  [pulp.value(curtail[t]) or 0 for t in hours],
        "SoC_kWh":     [pulp.value(soc[t])     or 0 for t in hours],
        "Price_EUR_MWh": prices_eur_mwh.values,
    })

    cost_optimised = float(pulp.value(prob.objective))

    # Baseline cost: solar only, no BESS
    base_net = np.maximum(load_kw.values - solar_kw.values, 0)
    cost_base = float(np.sum(base_net * prices_kwh))

    # Total savings vs grid-only (no solar, no BESS)
    savings = cost_grid_only - cost_optimised

    return {
        "status": status,
        "schedule": schedule,
        "cost_optimised": cost_optimised,
        "cost_base": cost_base,
        "cost_grid_only": cost_grid_only,
        "savings": savings,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5b – ENERGY FLOW DECOMPOSITION
# ═════════════════════════════════════════════════════════════════════════════

def compute_energy_flows(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process the LP schedule to decompose each hour's power flows into
    five labelled streams:

      Solar → Load      : solar directly serving the facility
      Solar → Battery   : solar excess stored in the battery
      Solar → Curtailed : solar that cannot be used or stored
      Grid  → Load      : grid directly serving the facility
      Grid  → Battery   : grid used to pre-charge the battery
      Battery → Load    : battery discharging to serve the facility

    All values are in kW (average over the hour, i.e. kWh if multiplied by 1h).
    """
    f = schedule.copy()

    solar_available = np.maximum(f["Solar_kW"].values, 0)
    curtailed       = np.maximum(f["Curtail_kW"].values, 0)
    solar_used      = np.maximum(solar_available - curtailed, 0)  # solar actually consumed

    load   = f["Load_kW"].values
    charge = f["Charge_kW"].values
    disch  = f["Disch_kW"].values
    grid   = f["Grid_kW"].values

    # Solar first serves load directly, remainder goes to battery
    solar_to_load = np.minimum(solar_used, load)
    solar_excess  = np.maximum(solar_used - solar_to_load, 0)
    solar_to_batt = np.minimum(charge, solar_excess)          # solar fraction of charging
    # Bound grid_to_batt to at most what the grid actually imported (prevents over-allocation)
    grid_to_batt  = np.minimum(np.maximum(charge - solar_to_batt, 0), grid)

    # Grid covers whatever load solar didn't + potential battery pre-charge
    grid_to_load  = np.maximum(grid - grid_to_batt, 0)

    # Battery discharging goes entirely to load
    batt_to_load  = disch

    f["Solar_to_Load"]  = solar_to_load
    f["Solar_to_Batt"]  = solar_to_batt
    f["Solar_Curtailed"]= curtailed
    f["Grid_to_Load"]   = grid_to_load
    f["Grid_to_Batt"]   = grid_to_batt
    f["Batt_to_Load"]   = batt_to_load

    return f


def make_sankey_hour(flows_df: pd.DataFrame, hour: int,
                     capacity_kwh: float) -> go.Figure:
    """
    Build a Plotly Sankey diagram for a single hour showing all power flows.
    Nodes: Solar Panel | Grid | Battery | Facility Load | Curtailed
    """
    row = flows_df[flows_df["Hour"] == hour].iloc[0]

    # Node indices:
    # 0 Solar Panel  1 Grid  2 Battery  3 Facility Load  4 Curtailed
    node_labels = ["Solar Panel", "Grid", "Battery", "Facility Load", "Curtailed"]
    node_colors = [C_BLUE, "#F5A623", C_GREEN, C_WHITE, "#555555"]

    sources, targets, values, link_colors = [], [], [], []

    def add(src, tgt, val, color):
        if val > 0.1:
            sources.append(src); targets.append(tgt)
            values.append(round(val, 2)); link_colors.append(color)

    add(0, 3, row["Solar_to_Load"],  "rgba(74,144,217,0.45)")   # Solar → Load
    add(0, 2, row["Solar_to_Batt"],  "rgba(38,178,75,0.55)")    # Solar → Battery
    add(0, 4, row["Solar_Curtailed"],"rgba(80,80,80,0.40)")     # Solar → Curtailed
    add(1, 3, row["Grid_to_Load"],   "rgba(245,166,35,0.45)")   # Grid → Load
    add(1, 2, row["Grid_to_Batt"],   "rgba(245,166,35,0.55)")   # Grid → Battery
    add(2, 3, row["Batt_to_Load"],   "rgba(38,178,75,0.70)")    # Battery → Load

    soc_pct = row["SoC_kWh"] / capacity_kwh * 100

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=22,
            label=node_labels,
            color=node_colors,
            line=dict(color=C_GREY2, width=1),
        ),
        link=dict(source=sources, target=targets,
                  value=values, color=link_colors),
    ))
    fig.update_layout(
        **plotly_base_layout(
            f"Hour {hour:02d}:00  |  Load {row['Load_kW']:.0f} kW  |  "
            f"Solar {row['Solar_kW']:.0f} kW  |  SoC {soc_pct:.0f}%"
        )
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 6a – LOGIN PAGE
# ═════════════════════════════════════════════════════════════════════════════

# Tiles displayed in scrambled visual order (tile_index, letter)
_LOGIN_TILES  = [(0,"B"),(1,"L"),(2,"T"),(3,"A"),(4,"I"),(5,"D"),(6,"E"),(7,"A")]
# Correct click sequence by tile_index → spells D-A-I-T-A-B-L-E
_LOGIN_TARGET = [5, 3, 4, 2, 7, 0, 1, 6]

_LOGIN_CSS = """
<style>
.dai-scatter {
    position: relative; height: 160px; width: 100%;
    overflow: visible; margin-bottom: 4px;
}
.dai-letter {
    position: absolute;
    font-size: 3.4rem; font-weight: 900; color: #26B24B;
    text-shadow: 0 0 18px rgba(38,178,75,0.7), 0 0 40px rgba(38,178,75,0.3);
    animation: daiFloat 3s ease-in-out infinite;
    user-select: none; pointer-events: none;
}
@keyframes daiFloat {
    0%,100% { transform: translateY(0px) rotate(-3deg) scale(1.0); }
    50%      { transform: translateY(-22px) rotate(3deg) scale(1.06); }
}
.dai-letter.l0 { left:  4%; top:42%; animation-delay:0.0s; }
.dai-letter.l1 { left: 16%; top:10%; animation-delay:0.4s; }
.dai-letter.l2 { left: 30%; top:58%; animation-delay:0.8s; }
.dai-letter.l3 { left: 43%; top:15%; animation-delay:1.2s; }
.dai-letter.l4 { left: 56%; top:52%; animation-delay:1.6s; }
.dai-letter.l5 { left: 68%; top:20%; animation-delay:2.0s; }
.dai-letter.l6 { left: 80%; top:46%; animation-delay:2.4s; }
.dai-letter.l7 { left: 91%; top: 8%; animation-delay:2.8s; }
.progress-track {
    font-size: 2.2rem; letter-spacing: 10px; text-align: center;
    font-weight: 700; padding: 10px 0 4px 0;
}
</style>
"""

_SCATTER_HTML = """
<div class="dai-scatter">
  <span class="dai-letter l0">B</span><span class="dai-letter l1">L</span>
  <span class="dai-letter l2">T</span><span class="dai-letter l3">A</span>
  <span class="dai-letter l4">I</span><span class="dai-letter l5">D</span>
  <span class="dai-letter l6">E</span><span class="dai-letter l7">A</span>
</div>
"""


def show_login_page():
    """Animated puzzle login screen — click scrambled tiles to spell DAITABLE."""
    if "click_seq" not in st.session_state:
        st.session_state["click_seq"] = []
    if "login_msg" not in st.session_state:
        st.session_state["login_msg"] = ""

    seq    = st.session_state["click_seq"]
    n_done = len(seq)

    # Completion check
    if n_done == 8:
        st.session_state["logged_in"] = True
        st.session_state.pop("click_seq", None)
        st.session_state.pop("login_msg", None)
        st.rerun()

    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)
    st.markdown(_SCATTER_HTML, unsafe_allow_html=True)

    st.markdown(
        f"<h2 style='text-align:center;color:{C_WHITE};margin-top:0;'>"
        "⚡ DAITABLE Energy Platform</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#AAAAAA;font-size:0.95rem;margin-top:-8px;'>"
        "Click the letters in the right order to spell the brand name</p>",
        unsafe_allow_html=True,
    )

    # Progress tracker
    target_word = "DAITABLE"
    done_str    = "".join(target_word[i] for i in range(n_done))
    todo_str    = "_" * (8 - n_done)
    st.markdown(
        f'<div class="progress-track">'
        f'<span style="color:#26B24B">{done_str}</span>'
        f'<span style="color:#444466">{todo_str}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.session_state["login_msg"]:
        st.markdown(
            f'<p style="text-align:center;color:#FF6666;font-size:0.9rem;">'
            f'{st.session_state["login_msg"]}</p>',
            unsafe_allow_html=True,
        )

    # Tile buttons
    cols = st.columns(8)
    clicked_tile = None
    for tile_idx, letter in _LOGIN_TILES:
        is_used = tile_idx in seq
        is_next = (n_done < 8 and _LOGIN_TARGET[n_done] == tile_idx)
        with cols[tile_idx]:
            if st.button(
                letter,
                key=f"tile_{tile_idx}",
                disabled=is_used,
                type="primary" if is_next else "secondary",
                use_container_width=True,
            ):
                clicked_tile = tile_idx

    if clicked_tile is not None:
        if clicked_tile == _LOGIN_TARGET[n_done]:
            st.session_state["click_seq"].append(clicked_tile)
            st.session_state["login_msg"] = ""
        else:
            st.session_state["click_seq"] = []
            st.session_state["login_msg"] = "✗ Wrong order — sequence reset! Try again 😅"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([3, 1, 3])
    with mid:
        if st.button("↺ Reset", key="login_reset", use_container_width=True):
            st.session_state["click_seq"] = []
            st.session_state["login_msg"] = ""
            st.rerun()

    st.divider()
    st.markdown(
        "<p style='text-align:center;color:#666688;font-size:0.85rem;'>"
        "Or enter the access code directly:</p>",
        unsafe_allow_html=True,
    )
    _, c, _ = st.columns([2, 2, 2])
    with c:
        pwd = st.text_input(
            "access code", type="password",
            label_visibility="collapsed",
            placeholder="access code…",
            key="pwd_fallback",
        )
    if pwd.strip().lower() == "daitable":
        st.session_state["logged_in"] = True
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 6b – MAIN UI & VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Login gate ─────────────────────────────────────────────────────
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if not st.session_state["logged_in"]:
        show_login_page()
        return

    # ── Header ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:8px;">
      <span style="font-size:2.2rem;">⚡</span>
      <div>
        <div style="font-size:1.6rem; font-weight:800; color:{C_WHITE};
                    letter-spacing:2px; line-height:1.1;">
          DAITABLE ENERGY OPTIMIZATION DASHBOARD
        </div>
        <div style="font-size:0.8rem; color:{C_GREEN}; letter-spacing:3px;">
          BESS + SOLAR PV PROFIT CALCULATOR
        </div>
      </div>
    </div>
    <hr style="border-color:{C_GREY2}; margin-bottom:20px;">
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f'<div style="color:{C_GREEN}; font-weight:700; '
                    f'font-size:1rem; letter-spacing:2px;">⚙ PARAMETERS</div>',
                    unsafe_allow_html=True)
        st.divider()

        st.markdown("**Data Sources**")
        real_csv       = st.text_input("Facility CSV path", value=DEFAULT_REAL_CSV)
        price_hist_csv = st.text_input("Price history CSV", value=DEFAULT_PRICE_HISTORY_CSV)
        # Default ref_date = day after price history ends (dynamic)
        _ref_default = datetime(2026, 3, 30).date()
        if os.path.exists(price_hist_csv):
            try:
                _ph_def      = load_real_price_history(price_hist_csv)
                _ref_default = (_ph_def.index.max() + timedelta(days=1)).date()
            except Exception:
                pass
        ref_date = st.date_input("Forecast reference date", value=_ref_default)
        st.divider()

        st.markdown("**Battery (BESS) Parameters**")
        cap_kwh     = st.slider("Capacity (kWh)", 50, 2000, 200, 50)
        max_pwr     = st.slider("Max Charge/Discharge Power (kW)", 10, 500, 50, 10)
        eff_pct     = st.slider("Round-trip Efficiency (%)", 70, 99, 90, 1)
        soc_min_pct = st.slider("Min SoC (%)", 5, 30, 10, 5) / 100
        soc_max_pct = st.slider("Max SoC (%)", 70, 100, 95, 5) / 100
        init_soc    = st.slider("Initial SoC (%)", 10, 100, 50, 5) / 100
        st.divider()

        st.markdown("**Solar PV Parameters**")
        pv_peak = st.slider("Peak Power (kWp)", 0, 500, 80, 10)
        pv_eff  = st.slider("Panel Efficiency (%)", 10, 25, 18, 1) / 100
        pv_lat  = st.slider("Site Latitude (°N)", 40, 55, 48, 1)
        st.divider()

        st.markdown("**System CapEx**")
        solar_cost_kwp = st.number_input("Solar cost (€/kWp)", min_value=0,
                                          value=700, step=50)
        batt_cost_kwh_input = st.number_input("Battery cost (€/kWh)", min_value=0,
                                               value=400, step=50)
        st.divider()

        st.markdown("**Forecasting**")
        forecast_horizon = st.slider("Forecast horizon (days)", 7, 60, 14)
        history_seed     = st.number_input("Synth-history seed", value=42, step=1)

        st.divider()
        if st.button("🔒 Log out", use_container_width=True):
            st.session_state["logged_in"] = False
            st.rerun()

    eff_one_way = (eff_pct / 100) ** 0.5

    # ── Tab layout ───────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "⚡  Optimization Dashboard",
        "📊  Real Data & Forecast",
        "💶  Cost Comparison",
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — Optimization Dashboard (real UVN data → Prophet → LP)
    # ════════════════════════════════════════════════════════════════════
    with tab1:
        # ── Load real facility data and train on it ───────────────────────
        real_data_ok = os.path.exists(real_csv)
        with st.spinner("Loading real facility data & training forecast model…"):
            if real_data_ok:
                facility_raw = load_real_facility_data(real_csv)
                # Build Prophet-compatible df from real Total_kW
                real_prophet_df = (
                    facility_raw[["Total_kW"]]
                    .rename(columns={"Total_kW": "y"})
                    .reset_index()
                    .rename(columns={facility_raw.index.name or "index": "ds"})
                )
                real_prophet_df["ds"] = pd.to_datetime(real_prophet_df["ds"])
                model, scaler, model_type = train_forecast_model(real_prophet_df)
                # Full real history for the history panel
                history_tail = facility_raw["Total_kW"]
            else:
                # Fallback: synthetic history
                history_df_synth = generate_historical_consumption(seed=int(history_seed))
                model, scaler, model_type = train_forecast_model(history_df_synth)
                history_tail = None

        with st.spinner("Running LP optimisation…"):
            ref_dt         = datetime(ref_date.year, ref_date.month, ref_date.day)
            load_forecast  = forecast_next_24h(model, scaler, model_type, ref_dt)
            solar_forecast = solar_generation_forecast(
                peak_kw=pv_peak, efficiency=pv_eff, latitude_deg=float(pv_lat),
                month=ref_date.month, day_noise_seed=ref_date.day,
            )
            # Use real price history + Prophet forecast for the selected date
            if os.path.exists(price_hist_csv):
                _ph = load_real_price_history(price_hist_csv)
                prices = forecast_prices_24h(_ph.to_json(), ref_dt.isoformat())
                _price_source = "real + forecast"
            else:
                prices = _synthetic_prices()
                _price_source = "synthetic"
            result = optimize_bess_dispatch(
                load_kw=load_forecast, solar_kw=solar_forecast,
                prices_eur_mwh=prices,
                capacity_kwh=cap_kwh, max_power_kw=max_pwr,
                charge_eff=eff_one_way, discharge_eff=eff_one_way,
                soc_min_pct=soc_min_pct, soc_max_pct=soc_max_pct,
                initial_soc_pct=init_soc,
            )

        sched = result["schedule"]
        hours = list(range(24))

        # ── Section 1: Data Forecasts ────────────────────────────────────
        st.markdown('<div class="section-header">01 — Data Forecasts</div>',
                    unsafe_allow_html=True)
        col_chart, col_hist = st.columns([3, 2])

        with col_chart:
            fig1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig1.add_trace(go.Bar(
                x=hours, y=prices.values, name="Price (EUR/MWh)",
                marker=dict(color=prices.values,
                            colorscale=[[0, C_GREY2], [1, C_GREEN]],
                            line=dict(color=C_GREEN, width=0.6)),
                opacity=0.55,
            ), secondary_y=True)
            fig1.add_trace(go.Scatter(
                x=hours, y=load_forecast.values, name="Load Forecast (kW)",
                line=dict(color=C_GREEN, width=2.5), mode="lines+markers",
                marker=dict(size=5),
            ), secondary_y=False)
            fig1.add_trace(go.Scatter(
                x=hours, y=solar_forecast.values, name="Solar Forecast (kW)",
                line=dict(color=C_BLUE, width=2, dash="dash"),
                fill="tozeroy", fillcolor="rgba(74,144,217,0.12)", mode="lines",
            ), secondary_y=False)
            layout1 = plotly_base_layout("24-Hour Forecast: Price / Load / Solar")
            layout1["yaxis"]["title"] = "Power (kW)"
            layout1["yaxis2"] = dict(title="Price (EUR/MWh)", gridcolor="#2A2A38",
                                     tickfont=dict(color="#AAAAAA"),
                                     overlaying="y", side="right")
            layout1["xaxis"]["title"] = "Hour of Day"
            fig1.update_layout(**layout1)
            st.plotly_chart(fig1, use_container_width=True)

        with col_hist:
            fig2 = go.Figure()
            if real_data_ok and history_tail is not None:
                data_end_ts = history_tail.index.max()
                # Forecast starts right after last real data point (not ref_date)
                # so the green line connects seamlessly to the historical trace
                fc_start_dt = data_end_ts + timedelta(hours=1)
                fut_ts = [fc_start_dt + timedelta(hours=h) for h in range(24)]

                # Full real data (all available history)
                fig2.add_trace(go.Scatter(
                    x=history_tail.index, y=history_tail.values,
                    name="Real data (UVN)", line=dict(color="#7777AA", width=1.2)))

                # Bridge gap between data end and forecast with a faint dotted line
                if fc_start_dt > data_end_ts + timedelta(hours=1):
                    fig2.add_trace(go.Scatter(
                        x=[data_end_ts, fut_ts[0]],
                        y=[history_tail.iloc[-1], load_forecast.iloc[0]],
                        name="Gap", line=dict(color="#444455", width=1, dash="dot"),
                        showlegend=False))
            else:
                # Fallback: use ref_date
                fut_ts = [ref_dt + timedelta(hours=h) for h in range(24)]

            fig2.add_trace(go.Scatter(
                x=fut_ts, y=load_forecast.values,
                name=f"Forecast (24h from {fut_ts[0].strftime('%b %d')})",
                line=dict(color=C_GREEN, width=2.5)))
            src_label = "Real UVN" if real_data_ok else "Synthetic"
            layout2 = plotly_base_layout(f"Consumption: {src_label} History → Forecast")
            layout2["xaxis"]["title"] = "Timestamp"
            layout2["yaxis"]["title"] = "Load (kW)"
            fig2.update_layout(**layout2)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f'<span style="background:{C_GREY2}; color:{C_GREEN}; '
                    f'border-radius:4px; padding:3px 10px; font-size:0.75rem;">'
                    f'Forecast model: {model_type}</span>'
                    f'&nbsp;&nbsp;<span style="background:{C_GREY2}; color:{C_AMBER}; '
                    f'border-radius:4px; padding:3px 10px; font-size:0.75rem;">'
                    f'Price source: {_price_source}</span>',
                    unsafe_allow_html=True)
        st.divider()

        # ── Section 2: BESS Performance ──────────────────────────────────
        st.markdown('<div class="section-header">02 — BESS Performance & Optimisation</div>',
                    unsafe_allow_html=True)

        if result["status"] not in ("Optimal", "Feasible") or sched is None:
            st.error(f"LP solver: {result['status']}. Adjust parameters and retry.")
        else:
            cost_base  = result["cost_base"]
            cost_opt   = result["cost_optimised"]
            savings    = result["savings"]
            savings_pct     = (savings / cost_base * 100) if cost_base > 0 else 0
            savings_per_kwh = savings / cap_kwh if cap_kwh > 0 else 0

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(kpi_card("Base Cost (no battery)", f"€{cost_base:.2f}",
                    "Grid cost without BESS"), unsafe_allow_html=True)
            with k2:
                st.markdown(kpi_card("Optimised Cost", f"€{cost_opt:.2f}",
                    "Grid cost with BESS"), unsafe_allow_html=True)
            with k3:
                st.markdown(kpi_card("Total Savings", f"€{savings:.2f}",
                    f"{savings_pct:.1f}% reduction", green=True), unsafe_allow_html=True)
            with k4:
                st.markdown(kpi_card("Savings per kWh Capacity",
                    f"€{savings_per_kwh:.4f}",
                    f"Per kWh of {cap_kwh} kWh installed"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            chart_l, chart_r = st.columns(2)

            with chart_l:
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(x=sched["Hour"], y=sched["Charge_kW"],
                    name="Charging (kW)", marker_color=C_BLUE))
                fig3.add_trace(go.Bar(x=sched["Hour"], y=-sched["Disch_kW"],
                    name="Discharging (kW)", marker_color=C_AMBER))
                fig3.add_trace(go.Scatter(x=sched["Hour"], y=sched["Grid_kW"],
                    name="Grid Import (kW)", line=dict(color=C_GREEN, width=2)))
                layout3 = plotly_base_layout("Battery Dispatch & Grid Import")
                layout3["barmode"] = "relative"
                layout3["xaxis"]["title"] = "Hour"
                layout3["yaxis"]["title"] = "Power (kW)"
                fig3.update_layout(**layout3)
                fig3.add_hline(y=0, line_color="#555", line_width=1)
                st.plotly_chart(fig3, use_container_width=True)

            with chart_r:
                soc_pct_arr = sched["SoC_kWh"] / cap_kwh * 100
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=sched["Hour"], y=soc_pct_arr,
                    name="State of Charge (%)", line=dict(color=C_GREEN, width=2.5),
                    fill="tozeroy", fillcolor="rgba(38,178,75,0.15)",
                    mode="lines+markers", marker=dict(size=6, color=C_GREEN)))
                fig4.add_hline(y=soc_min_pct * 100, line_dash="dot",
                    line_color=C_AMBER, annotation_text="Min SoC",
                    annotation_font_color=C_AMBER)
                fig4.add_hline(y=soc_max_pct * 100, line_dash="dot",
                    line_color=C_BLUE, annotation_text="Max SoC",
                    annotation_font_color=C_BLUE)
                layout4 = plotly_base_layout("Battery State of Charge (24h)")
                layout4["xaxis"]["title"] = "Hour"
                layout4["yaxis"]["title"] = "SoC (%)"
                layout4["yaxis"]["range"] = [0, 105]
                fig4.update_layout(**layout4)
                st.plotly_chart(fig4, use_container_width=True)

            # Energy flow area chart
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=sched["Hour"], y=sched["Load_kW"],
                name="Facility Load", line=dict(color=C_WHITE, width=1.5, dash="dot")))
            fig5.add_trace(go.Scatter(x=sched["Hour"], y=sched["Solar_kW"],
                name="Solar Generation", line=dict(color=C_BLUE, width=2),
                fill="tozeroy", fillcolor="rgba(74,144,217,0.18)"))
            fig5.add_trace(go.Scatter(x=sched["Hour"], y=sched["Grid_kW"],
                name="Grid Import (optimised)", line=dict(color=C_GREEN, width=2.5),
                fill="tozeroy", fillcolor="rgba(38,178,75,0.18)"))
            base_net = np.maximum(sched["Load_kW"].values - sched["Solar_kW"].values, 0)
            fig5.add_trace(go.Scatter(x=sched["Hour"], y=base_net,
                name="Grid Import (baseline)",
                line=dict(color="#AA4444", width=1.5, dash="dash")))
            layout5 = plotly_base_layout("Energy Flow: Load vs Solar vs Grid")
            layout5["xaxis"]["title"] = "Hour"
            layout5["yaxis"]["title"] = "Power (kW)"
            fig5.update_layout(**layout5)
            st.plotly_chart(fig5, use_container_width=True)
            st.divider()

            # ── Section 3: Solar Flow Explorer ───────────────────────────
            st.markdown('<div class="section-header">03 — Solar Charging Flow Explorer</div>',
                        unsafe_allow_html=True)
            flows = compute_energy_flows(sched)

            fig6 = go.Figure()
            flow_traces = [
                ("Solar_to_Load",  "Solar → Load",    C_BLUE,    "rgba(74,144,217,0.55)"),
                ("Solar_to_Batt",  "Solar → Battery", C_GREEN,   "rgba(38,178,75,0.55)"),
                ("Grid_to_Batt",   "Grid → Battery",  C_AMBER,   "rgba(245,166,35,0.45)"),
                ("Grid_to_Load",   "Grid → Load",     "#CC3333", "rgba(204,51,51,0.40)"),
                ("Batt_to_Load",   "Battery → Load",  "#A855F7", "rgba(168,85,247,0.50)"),
                ("Solar_Curtailed","Solar Curtailed",  "#555555", "rgba(80,80,80,0.35)"),
            ]
            for col_name, name, line_col, fill_col in flow_traces:
                fig6.add_trace(go.Scatter(
                    x=flows["Hour"], y=flows[col_name], name=name,
                    stackgroup="flows", line=dict(color=line_col, width=1.2),
                    fillcolor=fill_col, mode="lines",
                    hovertemplate=f"<b>{name}</b><br>Hour %{{x}}:00<br>%{{y:.1f}} kW<extra></extra>",
                ))
            layout6 = plotly_base_layout("24-Hour Energy Flow Decomposition (stacked)")
            layout6["xaxis"]["title"] = "Hour of Day"
            layout6["yaxis"]["title"] = "Power (kW)"
            fig6.update_layout(**layout6)
            st.plotly_chart(fig6, use_container_width=True)

            total_solar_gen    = flows["Solar_kW"].sum()
            total_s2b          = flows["Solar_to_Batt"].sum()
            total_s2l          = flows["Solar_to_Load"].sum()
            total_g2b          = flows["Grid_to_Batt"].sum()
            total_curtailed    = flows["Solar_Curtailed"].sum()
            solar_self_use_pct = ((total_s2l + total_s2b) / total_solar_gen * 100
                                   if total_solar_gen > 0 else 0)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(kpi_card("Solar → Battery", f"{total_s2b:.1f} kWh",
                    "Stored from PV today"), unsafe_allow_html=True)
            with m2:
                st.markdown(kpi_card("Solar → Load", f"{total_s2l:.1f} kWh",
                    "Direct PV self-consumption"), unsafe_allow_html=True)
            with m3:
                st.markdown(kpi_card("Grid → Battery", f"{total_g2b:.1f} kWh",
                    "Pre-charged from grid"), unsafe_allow_html=True)
            with m4:
                st.markdown(kpi_card("Solar Self-Use", f"{solar_self_use_pct:.0f}%",
                    f"{total_curtailed:.1f} kWh curtailed",
                    green=(solar_self_use_pct >= 70)), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div style="color:{C_WHITE}; font-size:0.9rem; font-weight:600;'
                        f'margin-bottom:8px;">Hour-by-Hour Flow Breakdown</div>',
                        unsafe_allow_html=True)
            default_hour  = int(flows["Solar_to_Batt"].idxmax()) if flows["Solar_to_Batt"].max() > 0 else 12
            selected_hour = st.slider("Scrub through forecast hours",
                min_value=0, max_value=23, value=default_hour, format="%d:00")
            row_f = flows[flows["Hour"] == selected_hour].iloc[0]
            soc_pct_sel = row_f["SoC_kWh"] / cap_kwh * 100
            action_parts = []
            if row_f["Solar_to_Batt"] > 1:
                action_parts.append(f"charging {row_f['Solar_to_Batt']:.0f} kW from solar")
            if row_f["Grid_to_Batt"] > 1:
                action_parts.append(f"charging {row_f['Grid_to_Batt']:.0f} kW from grid")
            if row_f["Batt_to_Load"] > 1:
                action_parts.append(f"discharging {row_f['Batt_to_Load']:.0f} kW to load")
            if row_f["Solar_Curtailed"] > 1:
                action_parts.append(f"curtailing {row_f['Solar_Curtailed']:.0f} kW solar")
            if not action_parts:
                action_parts = ["battery idle"]
            price_col = C_GREEN if row_f["Price_EUR_MWh"] < prices.median() else C_AMBER
            st.markdown(
                f'<div style="background:{C_GREY2}; border-radius:8px; padding:10px 16px;'
                f'margin-bottom:12px; font-size:0.85rem; color:{C_WHITE};">'
                f'<span style="color:{C_GREEN}; font-weight:700;">{selected_hour:02d}:00</span>'
                f'&nbsp;|&nbsp;Price: <span style="color:{price_col}; font-weight:700;">'
                f'{row_f["Price_EUR_MWh"]:.1f} EUR/MWh</span>'
                f'&nbsp;|&nbsp;SoC: <span style="color:{C_GREEN};">{soc_pct_sel:.0f}%</span>'
                f'&nbsp;|&nbsp;{" · ".join(action_parts)}</div>',
                unsafe_allow_html=True)
            st.plotly_chart(make_sankey_hour(flows, selected_hour, cap_kwh),
                            use_container_width=True)

            with st.expander("View Hourly Schedule Table"):
                display = sched.copy()
                display["Price_EUR_kWh"]  = (display["Price_EUR_MWh"] / 1000).round(4)
                display["HourlyCost_EUR"] = (display["Grid_kW"] * display["Price_EUR_kWh"]).round(4)
                display = display.drop(columns=["Price_EUR_MWh"]).round(2)
                st.dataframe(display.style.format("{:.2f}")
                    .background_gradient(subset=["SoC_kWh"], cmap="Greens")
                    .background_gradient(subset=["HourlyCost_EUR"], cmap="Reds"),
                    use_container_width=True)

        st.markdown(
            f'<div class="footer">DAITABLE brand guideline &nbsp;|&nbsp; Page 2 &nbsp;|&nbsp; '
            f'© {ref_date.year} DAITABLE &nbsp;|&nbsp; '
            f'LP status: <span style="color:{C_GREEN}">{result["status"]}</span></div>',
            unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — Real Facility Data & Forecast
    # ════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">Real Facility Data — UVN FA 1.1</div>',
                    unsafe_allow_html=True)

        if not os.path.exists(real_csv):
            st.error(f"Facility CSV not found: {real_csv}")
        else:
            with st.spinner("Loading facility data…"):
                facility_df = load_real_facility_data(real_csv)

            meas_cols = [c for c in facility_df.columns if c != "Total_kW"]
            data_start = facility_df.index.min()
            data_end   = facility_df.index.max()

            # ── Summary KPIs ──────────────────────────────────────────────
            total_kwh   = facility_df["Total_kW"].sum()        # kWh (1-h steps)
            mean_kw     = facility_df["Total_kW"].mean()
            peak_kw     = facility_df["Total_kW"].max()
            n_days      = (data_end - data_start).days + 1

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(kpi_card("Period", f"{n_days} days",
                    f"{data_start.date()} → {data_end.date()}"), unsafe_allow_html=True)
            with k2:
                st.markdown(kpi_card("Total Energy", f"{total_kwh:,.0f} kWh",
                    f"{total_kwh/n_days:.0f} kWh/day avg"), unsafe_allow_html=True)
            with k3:
                st.markdown(kpi_card("Avg Load", f"{mean_kw:.1f} kW",
                    "Hourly average"), unsafe_allow_html=True)
            with k4:
                st.markdown(kpi_card("Peak Load", f"{peak_kw:.1f} kW",
                    "Max hourly reading"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Stacked area by measurement ───────────────────────────────
            st.markdown('<div class="section-header">Consumption by System</div>',
                        unsafe_allow_html=True)
            fig_area = go.Figure()
            for m in meas_cols:
                label = MEAS_LABELS.get(m, m)
                color = MEAS_COLORS.get(m, C_WHITE)
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                fig_area.add_trace(go.Scatter(
                    x=facility_df.index, y=facility_df[m],
                    name=label, stackgroup="meas", mode="lines",
                    line=dict(color=color, width=0.8),
                    fillcolor=f"rgba({r},{g},{b},0.55)",
                    hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y:.1f}} kW<extra></extra>",
                ))
            lay_area = plotly_base_layout("Hourly Power by System (kW, stacked)")
            lay_area["xaxis"]["title"] = "Date"
            lay_area["yaxis"]["title"] = "Power (kW)"
            fig_area.update_layout(**lay_area)
            st.plotly_chart(fig_area, use_container_width=True)

            # ── Daily totals bar ──────────────────────────────────────────
            daily = facility_df["Total_kW"].resample("1D").sum()  # kWh/day
            fig_daily = go.Figure(go.Bar(
                x=daily.index, y=daily.values, name="Daily Energy (kWh)",
                marker=dict(
                    color=daily.values,
                    colorscale=[[0, C_GREY2], [0.5, C_GREEN], [1, C_AMBER]],
                    line=dict(color=C_GREY2, width=0.4),
                ),
            ))
            lay_daily = plotly_base_layout("Daily Energy Consumption (kWh)")
            lay_daily["xaxis"]["title"] = "Date"
            lay_daily["yaxis"]["title"] = "kWh/day"
            fig_daily.update_layout(**lay_daily)
            st.plotly_chart(fig_daily, use_container_width=True)

            # ── Diurnal heatmap ───────────────────────────────────────────
            pivot_hm = facility_df["Total_kW"].copy()
            pivot_hm.index = pd.to_datetime(pivot_hm.index)
            heat = pivot_hm.groupby([pivot_hm.index.date, pivot_hm.index.hour]).mean().unstack(level=1)
            fig_heat = go.Figure(go.Heatmap(
                z=heat.values, x=[f"{h:02d}:00" for h in range(24)],
                y=[str(d) for d in heat.index],
                colorscale=[[0, C_GREY2], [0.5, C_GREEN], [1, C_AMBER]],
                hovertemplate="Date: %{y}<br>Hour: %{x}<br>Load: %{z:.1f} kW<extra></extra>",
            ))
            lay_heat = plotly_base_layout("Load Heatmap — Date × Hour (kW)")
            lay_heat["xaxis"]["title"] = "Hour of Day"
            lay_heat["yaxis"]["title"] = "Date"
            fig_heat.update_layout(**lay_heat)
            st.plotly_chart(fig_heat, use_container_width=True)

            # ── Prophet forecast forward ──────────────────────────────────
            st.markdown('<div class="section-header">Load Forecast</div>',
                        unsafe_allow_html=True)
            with st.spinner(f"Training Prophet on real data & forecasting {forecast_horizon} days ahead…"):
                prophet_df = facility_df[["Total_kW"]].rename(columns={"Total_kW": "y"}).copy()
                prophet_df.index.name = "ds"
                prophet_df = prophet_df.reset_index()

                if PROPHET_AVAILABLE:
                    m_real = Prophet(daily_seasonality=True, weekly_seasonality=True,
                                     seasonality_mode="multiplicative",
                                     changepoint_prior_scale=0.15, interval_width=0.80)
                    m_real.fit(prophet_df)
                    fut = m_real.make_future_dataframe(periods=forecast_horizon * 24, freq="h")
                    fc  = m_real.predict(fut)
                else:
                    # Ridge fallback reusing Module 3 logic
                    m_real, sc_real, _ = train_forecast_model(
                        prophet_df.rename(columns={"ds": "ds", "y": "y"}))
                    fc = None  # simplified – skip if no Prophet

                fig_fc = go.Figure()
                # Historical
                fig_fc.add_trace(go.Scatter(
                    x=prophet_df["ds"], y=prophet_df["y"],
                    name="Actual (historical)", mode="lines",
                    line=dict(color="#555577", width=1),
                ))
                if PROPHET_AVAILABLE and fc is not None:
                    hist_end = prophet_df["ds"].max()
                    fc_fut   = fc[fc["ds"] > hist_end]
                    fc_hist  = fc[fc["ds"] <= hist_end]
                    # Fitted on history
                    fig_fc.add_trace(go.Scatter(
                        x=fc_hist["ds"], y=fc_hist["yhat"],
                        name="Fitted", line=dict(color=C_GREEN, width=1, dash="dot"),
                    ))
                    # Forecast band
                    fig_fc.add_trace(go.Scatter(
                        x=pd.concat([fc_fut["ds"], fc_fut["ds"].iloc[::-1]]).tolist(),
                        y=pd.concat([fc_fut["yhat_upper"], fc_fut["yhat_lower"].iloc[::-1]]).tolist(),
                        fill="toself", fillcolor="rgba(38,178,75,0.15)",
                        line=dict(color="rgba(0,0,0,0)"), name="80% CI",
                        showlegend=True,
                    ))
                    fig_fc.add_trace(go.Scatter(
                        x=fc_fut["ds"], y=fc_fut["yhat"],
                        name=f"Forecast (+{forecast_horizon}d)",
                        line=dict(color=C_GREEN, width=2.5),
                    ))
                lay_fc = plotly_base_layout(f"Real Data + {forecast_horizon}-Day Load Forecast")
                lay_fc["xaxis"]["title"] = "Date"
                lay_fc["yaxis"]["title"] = "Load (kW)"
                fig_fc.update_layout(**lay_fc)
                # Vertical line at data end — pass as millisecond timestamp (Plotly requirement)
                data_end_ms = int(data_end.timestamp() * 1000)
                fig_fc.add_vline(x=data_end_ms, line_dash="dash",
                                 line_color=C_AMBER, annotation_text="Data end",
                                 annotation_font_color=C_AMBER)
                st.plotly_chart(fig_fc, use_container_width=True)

            # ── Price history + forecast section ─────────────────────────
            st.markdown('<div class="section-header">Electricity Price History & Forecast</div>',
                        unsafe_allow_html=True)

            if os.path.exists(price_hist_csv):
                with st.spinner("Loading price history & forecasting…"):
                    ph = load_real_price_history(price_hist_csv)
                    ph_end    = ph.index.max()
                    ph_end_ms = int(ph_end.timestamp() * 1000)

                    # Forecast from ph_end up to at least ref_date + 3 days
                    ref_dt_naive = datetime(ref_date.year, ref_date.month, ref_date.day)
                    fc_days = max(7, (ref_dt_naive.date() - ph_end.date()).days + 3)
                    ph_json  = ph.to_json()   # serialise once, reuse in loop
                    fc_parts = []
                    for d in range(fc_days):
                        day_dt  = ph_end + timedelta(days=d + 1)
                        day_fc  = forecast_prices_24h(ph_json, day_dt.strftime("%Y-%m-%d"))
                        # Re-index from 0..23 to actual timestamps
                        day_ts  = pd.date_range(day_dt.normalize(), periods=24, freq="1h")
                        fc_parts.append(pd.Series(day_fc.values, index=day_ts))
                    fc_series = pd.concat(fc_parts)
                    fc_series.name = "Price_EUR_MWh"

                # Daily stats for ribbon
                daily_min  = ph.resample("1D").min()
                daily_max  = ph.resample("1D").max()
                daily_mean = ph.resample("1D").mean()

                fig_ph = go.Figure()
                # Min-max ribbon
                fig_ph.add_trace(go.Scatter(
                    x=pd.concat([daily_min.index.to_series(),
                                 daily_max.index.to_series().iloc[::-1]]).tolist(),
                    y=pd.concat([daily_min, daily_max.iloc[::-1]]).tolist(),
                    fill="toself", fillcolor="rgba(38,178,75,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Daily min-max range", showlegend=True,
                ))
                # Hourly real prices (thin)
                fig_ph.add_trace(go.Scatter(
                    x=ph.index, y=ph.values,
                    name="Real price (hourly)", mode="lines",
                    line=dict(color="#666688", width=0.8),
                ))
                # Daily mean
                fig_ph.add_trace(go.Scatter(
                    x=daily_mean.index, y=daily_mean.values,
                    name="Daily mean", mode="lines",
                    line=dict(color=C_WHITE, width=1.8),
                ))
                # Forecast
                fig_ph.add_trace(go.Scatter(
                    x=fc_series.index, y=fc_series.values,
                    name=f"Price forecast (+{fc_days}d)",
                    mode="lines",
                    line=dict(color=C_GREEN, width=2.5, dash="dash"),
                ))
                lay_ph = plotly_base_layout("Real Electricity Price History + Forecast (EUR/MWh)")
                lay_ph["xaxis"]["title"] = "Date"
                lay_ph["yaxis"]["title"] = "EUR/MWh"
                fig_ph.update_layout(**lay_ph)
                fig_ph.add_vline(x=ph_end_ms, line_dash="dot", line_color=C_AMBER,
                                 annotation_text="History end",
                                 annotation_font_color=C_AMBER)
                ref_dt_ms = int(ref_dt_naive.timestamp() * 1000)
                fig_ph.add_vline(x=ref_dt_ms, line_dash="dash", line_color=C_GREEN,
                                 annotation_text=f"Ref: {ref_date}",
                                 annotation_font_color=C_GREEN)
                st.plotly_chart(fig_ph, use_container_width=True)

                # Quick price KPIs
                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    st.markdown(kpi_card("Avg Price", f"{ph.mean():.1f} €/MWh",
                        f"{ph.index.min().date()} – {ph_end.date()}"), unsafe_allow_html=True)
                with p2:
                    st.markdown(kpi_card("Peak Price", f"{ph.max():.1f} €/MWh",
                        f"at {ph.idxmax().strftime('%b %d %H:%M')}"), unsafe_allow_html=True)
                with p3:
                    neg_hours = (ph < 0).sum()
                    st.markdown(kpi_card("Negative Price Hours", str(int(neg_hours)),
                        f"Min: {ph.min():.1f} €/MWh"), unsafe_allow_html=True)
                with p4:
                    fc_mean = fc_series.mean()
                    fc_col  = C_GREEN if fc_mean < ph.mean() else C_AMBER
                    st.markdown(kpi_card("Forecast avg (next 5d)",
                        f"{fc_mean:.1f} €/MWh",
                        "vs " + (f"↓ {ph.mean()-fc_mean:.1f} cheaper" if fc_mean < ph.mean()
                                 else f"↑ {fc_mean-ph.mean():.1f} higher"),
                        green=(fc_mean < ph.mean())), unsafe_allow_html=True)
            else:
                st.info(f"Price history CSV not found: {price_hist_csv}")

        st.markdown(
            f'<div class="footer">DAITABLE brand guideline &nbsp;|&nbsp; Page 2 &nbsp;|&nbsp;'
            f'© 2026 DAITABLE</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — Cost Comparison: Reality vs. BESS + Solar
    # ════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">💶 Cost Comparison — Reality vs. BESS + Solar</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:#AAAAAA; font-size:0.85rem; margin-bottom:16px;">'
            f'Day-by-day LP simulation over the entire real dataset. '
            f'Each day is optimised independently with the BESS + Solar parameters from the sidebar.'
            f'</div>', unsafe_allow_html=True)

        if not os.path.exists(real_csv):
            st.error(f"Facility CSV not found: {real_csv}")
        else:
            with st.spinner("Loading data & generating price series…"):
                facility_df2  = load_real_facility_data(real_csv)
                hourly_load   = facility_df2["Total_kW"]
                hourly_prices = generate_price_series_range(
                    hourly_load.index, price_history_csv=price_hist_csv, seed=7)

            with st.spinner(f"Running LP for {len(hourly_load.index.normalize().unique())} days… (cached after first run)"):
                sim_df = run_period_simulation(
                    load_json=hourly_load.to_json(),
                    price_json=hourly_prices.to_json(),
                    capacity_kwh=cap_kwh,
                    max_power_kw=max_pwr,
                    charge_eff=eff_one_way,
                    discharge_eff=eff_one_way,
                    soc_min_pct=soc_min_pct,
                    soc_max_pct=soc_max_pct,
                    pv_peak_kw=float(pv_peak),
                    pv_efficiency=pv_eff,
                    latitude=float(pv_lat),
                )

            if sim_df.empty:
                st.error("Simulation returned no results. Check data and parameters.")
            else:
                sim_df["Date"] = pd.to_datetime(sim_df["Date"])
                sim_df["Cumulative_GridOnly_EUR"]   = sim_df["GridOnly_Cost_EUR"].cumsum()
                sim_df["Cumulative_Baseline_EUR"]   = sim_df["Baseline_Cost_EUR"].cumsum()
                sim_df["Cumulative_Optimised_EUR"]  = sim_df["Optimised_Cost_EUR"].cumsum()
                sim_df["Cumulative_Savings_EUR"]    = sim_df["Savings_EUR"].cumsum()
                sim_df["Savings_pct"]               = (
                    sim_df["Savings_EUR"] / sim_df["GridOnly_Cost_EUR"] * 100
                ).clip(0, 100)

                total_grid_only = sim_df["GridOnly_Cost_EUR"].sum()
                total_baseline  = sim_df["Baseline_Cost_EUR"].sum()
                total_optimised = sim_df["Optimised_Cost_EUR"].sum()
                total_savings   = sim_df["Savings_EUR"].sum()   # vs grid-only
                total_load_kwh  = sim_df["Total_Load_kWh"].sum()
                total_solar_kwh = sim_df["Total_Solar_kWh"].sum()
                avg_savings_pct = (total_savings / total_grid_only * 100) if total_grid_only > 0 else 0
                solar_only_savings = total_grid_only - total_baseline
                bess_only_savings  = total_baseline - total_optimised
                n_days_sim      = len(sim_df)
                annual_savings  = total_savings / n_days_sim * 365

                # ── Top KPIs ──────────────────────────────────────────────
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                with k1:
                    st.markdown(kpi_card("Grid Only Cost",
                        f"€{total_grid_only:,.2f}",
                        "No Solar, no BESS"), unsafe_allow_html=True)
                with k2:
                    st.markdown(kpi_card("Solar Only Cost",
                        f"€{total_baseline:,.2f}",
                        f"Saves €{solar_only_savings:,.0f} from solar"),
                        unsafe_allow_html=True)
                with k3:
                    st.markdown(kpi_card("Optimised Cost",
                        f"€{total_optimised:,.2f}",
                        f"+€{bess_only_savings:,.0f} from BESS"),
                        unsafe_allow_html=True)
                with k4:
                    st.markdown(kpi_card("Total Savings",
                        f"€{total_savings:,.2f}",
                        f"{avg_savings_pct:.1f}% vs grid-only", green=True),
                        unsafe_allow_html=True)
                with k5:
                    st.markdown(kpi_card("Est. Annual Savings",
                        f"€{annual_savings:,.0f}",
                        "Extrapolated from period"), unsafe_allow_html=True)
                with k6:
                    total_capex = cap_kwh * batt_cost_kwh_input + pv_peak * solar_cost_kwp
                    payback = total_capex / annual_savings if annual_savings > 0 else 0
                    st.markdown(kpi_card("Est. Payback",
                        f"{payback:.1f} yrs",
                        f"CapEx €{total_capex:,.0f} total"), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Daily cost comparison bars ────────────────────────────
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    x=sim_df["Date"], y=sim_df["GridOnly_Cost_EUR"],
                    name="Grid Only (€)", marker_color="#AA3333", opacity=0.7,
                ))
                fig_cmp.add_trace(go.Bar(
                    x=sim_df["Date"], y=sim_df["Baseline_Cost_EUR"],
                    name="Solar Only (€)", marker_color=C_AMBER, opacity=0.8,
                ))
                fig_cmp.add_trace(go.Bar(
                    x=sim_df["Date"], y=sim_df["Optimised_Cost_EUR"],
                    name="Solar + BESS (€)", marker_color=C_GREEN, opacity=0.85,
                ))
                lay_cmp = plotly_base_layout("Daily Cost: Grid Only vs. Solar Only vs. Solar+BESS")
                lay_cmp["barmode"]         = "group"
                lay_cmp["xaxis"]["title"]  = "Date"
                lay_cmp["yaxis"]["title"]  = "Cost (€)"
                fig_cmp.update_layout(**lay_cmp)
                st.plotly_chart(fig_cmp, use_container_width=True)

                # ── Cumulative savings & daily savings% side-by-side ──────
                col_cum, col_pct = st.columns(2)

                with col_cum:
                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=sim_df["Date"], y=sim_df["Cumulative_GridOnly_EUR"],
                        name="Grid Only", fill="tozeroy",
                        fillcolor="rgba(170,51,51,0.10)",
                        line=dict(color="#AA3333", width=1.5, dash="dot"),
                    ))
                    fig_cum.add_trace(go.Scatter(
                        x=sim_df["Date"], y=sim_df["Cumulative_Baseline_EUR"],
                        name="Solar Only",
                        line=dict(color=C_AMBER, width=2),
                    ))
                    fig_cum.add_trace(go.Scatter(
                        x=sim_df["Date"], y=sim_df["Cumulative_Optimised_EUR"],
                        name="Solar + BESS", fill="tozeroy",
                        fillcolor="rgba(38,178,75,0.15)",
                        line=dict(color=C_GREEN, width=2.5),
                    ))
                    fig_cum.add_trace(go.Scatter(
                        x=sim_df["Date"], y=sim_df["Cumulative_Savings_EUR"],
                        name="Cumulative Savings (vs Grid Only)",
                        line=dict(color="#FFFFFF", width=2, dash="dash"),
                    ))
                    lay_cum = plotly_base_layout("Cumulative Costs & Savings (€)")
                    lay_cum["xaxis"]["title"] = "Date"
                    lay_cum["yaxis"]["title"] = "€"
                    fig_cum.update_layout(**lay_cum)
                    st.plotly_chart(fig_cum, use_container_width=True)

                with col_pct:
                    fig_pct = go.Figure()
                    fig_pct.add_trace(go.Bar(
                        x=sim_df["Date"], y=sim_df["Savings_pct"],
                        name="Daily Savings %",
                        marker=dict(
                            color=sim_df["Savings_pct"],
                            colorscale=[[0, C_GREY2], [0.5, C_GREEN], [1, C_AMBER]],
                        ),
                    ))
                    lay_pct = plotly_base_layout("Daily Savings (% of grid-only cost)")
                    lay_pct["xaxis"]["title"] = "Date"
                    lay_pct["yaxis"]["title"] = "Savings %"
                    fig_pct.update_layout(**lay_pct)
                    st.plotly_chart(fig_pct, use_container_width=True)

                # ── Solar contribution summary ────────────────────────────
                st.divider()
                st.markdown('<div class="section-header">Solar Contribution</div>',
                            unsafe_allow_html=True)
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown(kpi_card("Total Solar Generated",
                        f"{total_solar_kwh:,.0f} kWh",
                        f"Over {n_days_sim} days"), unsafe_allow_html=True)
                with s2:
                    solar_cover = total_solar_kwh / total_load_kwh * 100 if total_load_kwh > 0 else 0
                    st.markdown(kpi_card("Solar Coverage",
                        f"{solar_cover:.1f}%", "Of total facility load",
                        green=(solar_cover >= 20)), unsafe_allow_html=True)
                with s3:
                    st.markdown(kpi_card("Total Load",
                        f"{total_load_kwh:,.0f} kWh",
                        f"{total_load_kwh/n_days_sim:.0f} kWh/day avg"), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Detailed table ────────────────────────────────────────
                with st.expander("View Daily Simulation Table"):
                    disp = sim_df.copy()
                    disp["Date"] = disp["Date"].dt.date
                    for col_num in ["GridOnly_Cost_EUR", "Baseline_Cost_EUR",
                                    "Optimised_Cost_EUR",
                                    "Savings_EUR", "Cumulative_Savings_EUR"]:
                        disp[col_num] = disp[col_num].round(3)
                    disp["Savings_pct"] = disp["Savings_pct"].round(1)
                    st.dataframe(
                        disp[["Date", "Total_Load_kWh", "Total_Solar_kWh",
                              "GridOnly_Cost_EUR", "Baseline_Cost_EUR",
                              "Optimised_Cost_EUR",
                              "Savings_EUR", "Savings_pct",
                              "Cumulative_Savings_EUR", "LP_Status"]]
                        .style.format({
                            "Total_Load_kWh":      "{:.1f}",
                            "Total_Solar_kWh":     "{:.1f}",
                            "GridOnly_Cost_EUR":   "€{:.3f}",
                            "Baseline_Cost_EUR":   "€{:.3f}",
                            "Optimised_Cost_EUR":  "€{:.3f}",
                            "Savings_EUR":         "€{:.3f}",
                            "Savings_pct":         "{:.1f}%",
                            "Cumulative_Savings_EUR": "€{:.2f}",
                        })
                        .background_gradient(subset=["Savings_EUR"], cmap="Greens")
                        .background_gradient(subset=["Savings_pct"], cmap="Greens"),
                        use_container_width=True,
                    )

        st.markdown(
            f'<div class="footer">DAITABLE brand guideline &nbsp;|&nbsp; Page 2 &nbsp;|&nbsp;'
            f'© 2026 DAITABLE &nbsp;|&nbsp; Prices: synthetic OKTE-style curve</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    main()
