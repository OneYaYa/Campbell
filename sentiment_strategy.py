from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

@dataclass
class SentimentDirectionalConfig:
    # sentiment combination weights
    w1: float = 0.2
    w3: float = 0.3
    w7: float = 0.5

    # rolling z-score window for sentiment signal
    z_window: int = 120
    z_min_periods: int = 60

    # entry/exit thresholds (hysteresis)
    entry_z: float = 0.5
    exit_z: float = 0.1

    # volatility targeting
    target_vol: float = 0.01  # 1% daily vol target
    w_max: float = 2.0

    # cost model
    cost_per_unit: float = 0.0002  # 2 bps per unit turnover

    # safety filter
    enable_vol_spike_filter: bool = True
    vol_spike_mult: float = 2.5

    # execution convention
    next_day_execution: bool = True


# ============================================================
# Utilities
# ============================================================

def _required_cols_check(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _rolling_zscore(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    mu = x.rolling(window=window, min_periods=min_periods).mean()
    sd = x.rolling(window=window, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    return (x - mu) / sd


def load_yahoo_oil_csv(
    csv_path: str,
    *,
    date_col_in_csv: str = "Date",
    recompute_returns: bool = True,
) -> pd.DataFrame:
    """
    Loads your Yahoo-style CSV and standardizes columns to:
      date, open, high, low, close, volume, ret
    """
    df = pd.read_csv(csv_path)

    if date_col_in_csv not in df.columns:
        raise ValueError(f"Expected date column '{date_col_in_csv}' in {csv_path}, got {list(df.columns)}")

    df = df.rename(columns={date_col_in_csv: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # If your CSV already contains ret, you can set recompute_returns=False
    if recompute_returns:
        if "close" not in df.columns:
            raise ValueError("Cannot recompute returns without 'close' column.")
        df["ret"] = df["close"].pct_change()

    return df


def load_sentiment_csv(
    csv_path: str,
    *,
    date_col_in_csv: str = "published",
    sentiment_col: str = "sentiment",
    confidence_col: str = "confidence",
) -> pd.DataFrame:
    """
    Loads sentiment analysis CSV and aggregates by date.
    
    Input columns expected:
      - published (or custom date_col_in_csv): date column
      - sentiment (or custom sentiment_col): sentiment label ("Positive", "Negative", "Neutral")
      - confidence (or custom confidence_col): confidence score
    
    Output aggregates daily sentiment:
      - date: trading date
      - oil_sent_1d: mean sentiment for that day (-1=negative, 0=neutral, 1=positive)
      - oil_news_count_7d: rolling 7-day count of articles
    """
    df = pd.read_csv(csv_path)
    
    # Rename published column to date
    if date_col_in_csv not in df.columns:
        raise ValueError(f"Expected date column '{date_col_in_csv}' in CSV, got {list(df.columns)}")
    
    if sentiment_col not in df.columns:
        raise ValueError(f"Expected sentiment column '{sentiment_col}' in CSV")
    
    if confidence_col not in df.columns:
        raise ValueError(f"Expected confidence column '{confidence_col}' in CSV")
    
    df_work = df.copy()
    df_work[date_col_in_csv] = pd.to_datetime(df_work[date_col_in_csv])
    df_work["date"] = df_work[date_col_in_csv].dt.floor('D')  # Floor to trading day
    
    # Convert sentiment labels to numeric scores
    sentiment_map = {
        "Positive": 1.0,
        "Neutral": 0.0,
        "Negative": -1.0,
    }
    df_work["sent_score"] = df_work[sentiment_col].map(sentiment_map)
    df_work["weighted_sent"] = df_work["sent_score"] * df_work[confidence_col]
    
    # Daily aggregation
    daily_agg = df_work.groupby("date").agg({
        "weighted_sent": "mean",
    }).reset_index()
    daily_agg.rename(columns={"weighted_sent": "oil_sent_1d"}, inplace=True)
    
    # Count articles per day
    daily_counts = df_work.groupby("date").size().reset_index(name="daily_count")
    daily_agg = daily_agg.merge(daily_counts, on="date")
    
    # Rolling 7-day article count
    daily_agg = daily_agg.sort_values("date").reset_index(drop=True)
    daily_agg["oil_news_count_7d"] = daily_agg["daily_count"].rolling(
        window=7, min_periods=1
    ).sum().astype(int)
    
    # Compute rolling 3d and 7d sentiment
    daily_agg["oil_sent_3d"] = daily_agg["oil_sent_1d"].rolling(
        window=3, min_periods=1
    ).mean()
    daily_agg["oil_sent_7d"] = daily_agg["oil_sent_1d"].rolling(
        window=7, min_periods=1
    ).mean()
    
    # Return only needed columns
    return daily_agg[["date", "oil_sent_1d", "oil_sent_3d", "oil_sent_7d", "oil_news_count_7d"]]


def add_rolling_volatility(
    df: pd.DataFrame,
    *,
    ret_col: str = "ret",
    vol20_window: int = 20,
    vol60_window: int = 60,
    min_periods20: Optional[int] = None,
    min_periods60: Optional[int] = None,
) -> pd.DataFrame:
    out = df.copy()
    _required_cols_check(out, [ret_col])

    mp20 = vol20_window if min_periods20 is None else min_periods20
    mp60 = vol60_window if min_periods60 is None else min_periods60

    out["vol20"] = out[ret_col].rolling(vol20_window, min_periods=mp20).std(ddof=0)
    out["vol60"] = out[ret_col].rolling(vol60_window, min_periods=mp60).std(ddof=0)
    return out


# ============================================================
# Random sentiment generator (temporary placeholder)
# ============================================================

def generate_random_oil_sentiment_features(
    dates: pd.Series,
    *,
    seed: int = 7,
    phi: float = 0.92,
    shock_scale: float = 0.20,
    base_news_lambda: float = 18.0,
) -> pd.DataFrame:
    """
    Generates a *realistic-ish* daily sentiment process:
    - AR(1) latent sentiment (slow decay, good for oil narratives)
    - rolling means to get 1d/3d/7d
    - Poisson news counts (with mild co-movement with sentiment magnitude)

    Output columns:
      date, oil_sent_1d, oil_sent_3d, oil_sent_7d, oil_news_count_7d
    """
    d = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    n = len(d)

    rng = np.random.default_rng(seed)

    # AR(1) latent process with occasional shocks
    eps = rng.normal(0.0, shock_scale, size=n)
    latent = np.zeros(n, dtype=float)
    for i in range(1, n):
        latent[i] = phi * latent[i - 1] + eps[i]

    # squash to [-1, 1] like sentiment
    latent = np.tanh(latent)

    s = pd.Series(latent)

    oil_sent_1d = s
    oil_sent_3d = s.rolling(3, min_periods=1).mean()
    oil_sent_7d = s.rolling(7, min_periods=1).mean()

    # News counts: higher around high |sentiment| days
    intensity = np.clip(np.abs(oil_sent_7d.to_numpy()) * 10.0, 0.0, 10.0)
    lam = np.clip(base_news_lambda + intensity, 1.0, None)
    news_count_daily = rng.poisson(lam=lam, size=n).astype(int)

    oil_news_count_7d = pd.Series(news_count_daily).rolling(7, min_periods=1).sum().astype(int)

    out = pd.DataFrame(
        {
            "date": d,
            "oil_sent_1d": oil_sent_1d.to_numpy(),
            "oil_sent_3d": oil_sent_3d.to_numpy(),
            "oil_sent_7d": oil_sent_7d.to_numpy(),
            "oil_news_count_7d": oil_news_count_7d.to_numpy(),
        }
    )
    return out


def merge_sentiment_features(market_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    m = market_df.copy()
    s = sentiment_df.copy()
    m["date"] = pd.to_datetime(m["date"])
    s["date"] = pd.to_datetime(s["date"])
    out = m.merge(s, on="date", how="left")
    return out


# ============================================================
# Strategy logic
# ============================================================

def build_sentiment_signal(
    df: pd.DataFrame,
    cfg: SentimentDirectionalConfig,
    s1_col: str = "oil_sent_1d",
    s3_col: str = "oil_sent_3d",
    s7_col: str = "oil_sent_7d",
    count7_col: str = "oil_news_count_7d",
) -> pd.DataFrame:
    _required_cols_check(df, [s1_col, s3_col, s7_col, count7_col])

    out = df.copy()

    S_raw = cfg.w1 * out[s1_col] + cfg.w3 * out[s3_col] + cfg.w7 * out[s7_col]
    intensity = np.log1p(out[count7_col].clip(lower=0))
    out["S_raw"] = S_raw
    out["S_intensity"] = S_raw * intensity
    out["Z"] = _rolling_zscore(out["S_intensity"], window=cfg.z_window, min_periods=cfg.z_min_periods)

    return out


def run_directional_strategy(
    df: pd.DataFrame,
    cfg: SentimentDirectionalConfig,
    *,
    date_col: str = "date",
    ret_col: str = "ret",
    vol20_col: str = "vol20",
    vol60_col: str = "vol60",
) -> pd.DataFrame:
    req = [date_col, ret_col, vol20_col]
    if cfg.enable_vol_spike_filter:
        req.append(vol60_col)
    _required_cols_check(df, req)

    d0 = df.copy()
    d0[date_col] = pd.to_datetime(d0[date_col])
    d0 = d0.sort_values(date_col).reset_index(drop=True)

    d1 = build_sentiment_signal(d0, cfg)

    if cfg.enable_vol_spike_filter:
        vol_ok = d1[vol20_col] <= cfg.vol_spike_mult * d1[vol60_col]
    else:
        vol_ok = pd.Series(True, index=d1.index)

    # state machine
    pos_state = np.zeros(len(d1), dtype=np.int8)  # -1/0/+1
    weight = np.zeros(len(d1), dtype=float)

    prev_state = 0
    Z = d1["Z"].to_numpy(dtype=float)
    vol20 = d1[vol20_col].to_numpy(dtype=float)
    vol_ok_np = vol_ok.to_numpy(dtype=bool)

    for i in range(len(d1)):
        z = Z[i]
        state = prev_state

        if not np.isnan(z):
            if prev_state == 0:
                if z > cfg.entry_z and vol_ok_np[i]:
                    state = +1
                elif z < -cfg.entry_z and vol_ok_np[i]:
                    state = -1
            elif prev_state == +1:
                if z < cfg.exit_z:
                    state = 0
            elif prev_state == -1:
                if z > -cfg.exit_z:
                    state = 0

        if state == 0 or np.isnan(vol20[i]) or vol20[i] <= 0:
            w = 0.0
        else:
            w = float(state) * min(cfg.target_vol / float(vol20[i]), cfg.w_max)

        pos_state[i] = state
        weight[i] = w
        prev_state = state

    d1["pos_state"] = pos_state
    d1["weight"] = weight

    # costs
    d1["turnover"] = d1["weight"].diff().abs().fillna(d1["weight"].abs())
    d1["cost"] = d1["turnover"] * cfg.cost_per_unit

    # execution
    if cfg.next_day_execution:
        d1["weight_applied"] = d1["weight"].shift(1).fillna(0.0)
        d1["cost_applied"] = d1["cost"].shift(1).fillna(0.0)
    else:
        d1["weight_applied"] = d1["weight"]
        d1["cost_applied"] = d1["cost"]

    d1["strat_ret"] = d1["weight_applied"] * d1[ret_col] - d1["cost_applied"]
    d1["equity"] = (1.0 + d1["strat_ret"].fillna(0.0)).cumprod()

    return d1


def performance_summary(strat_df: pd.DataFrame, strat_ret_col: str = "strat_ret") -> dict:
    r = strat_df[strat_ret_col].dropna().astype(float)
    if r.empty:
        return {"error": "No returns to summarize."}

    ann = 252.0
    mu = r.mean()
    sd = r.std(ddof=0)
    sharpe = (mu / sd) * np.sqrt(ann) if sd > 0 else np.nan

    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0

    total_ret = float(equity.iloc[-1] - 1.0)
    ann_ret = float((1.0 + total_ret) ** (ann / len(r)) - 1.0) if len(r) > 0 else np.nan
    ann_vol = float(sd * np.sqrt(ann))

    return {
        "n_days": int(len(r)),
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": float(dd.min()),
        "avg_daily_turnover": float(strat_df["turnover"].mean()) if "turnover" in strat_df else np.nan,
        "avg_daily_cost": float(strat_df["cost"].mean()) if "cost" in strat_df else np.nan,
    }


# ============================================================
# High-level convenience: run end-to-end for one instrument
# ============================================================

def build_and_backtest_one_instrument(
    market_csv_path: str,
    *,
    seed: int,
    cfg: Optional[SentimentDirectionalConfig] = None,
    sentiment_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Loads market CSV and sentiment data (either from real CSV or generated randomly).
    
    Args:
        market_csv_path: Path to market data CSV (OHLCV)
        seed: Random seed for mock sentiment generation (if sentiment_csv is None)
        cfg: Strategy config
        sentiment_csv: Path to sentiment analysis CSV output (optional)
                      If provided, loads real sentiment; otherwise generates random data
    
    Returns:
        (backtest_df, performance_stats)
    """
    if cfg is None:
        cfg = SentimentDirectionalConfig()

    mkt = load_yahoo_oil_csv(market_csv_path, recompute_returns=True)
    mkt = add_rolling_volatility(mkt)

    # Load sentiment data
    if sentiment_csv:
        try:
            sent = load_sentiment_csv(sentiment_csv)
            print(f"Loaded sentiment from: {sentiment_csv}")
        except Exception as e:
            print(f"Error loading sentiment CSV {sentiment_csv}: {e}")
            print("Falling back to random sentiment generation...")
            sent = generate_random_oil_sentiment_features(mkt["date"], seed=seed)
    else:
        sent = generate_random_oil_sentiment_features(mkt["date"], seed=seed)

    df = merge_sentiment_features(mkt, sent)

    bt = run_directional_strategy(df, cfg)
    stats = performance_summary(bt)
    return bt, stats
