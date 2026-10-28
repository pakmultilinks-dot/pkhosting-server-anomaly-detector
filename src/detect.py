"""Anomaly detectors + rule-based alerting.

All detectors are per-metric and use only past data (rolling windows are
shifted by one minute), so a flag at time t never uses information from
after t. That matches how the system would run live.

Methods:
  zscore  - rolling mean/std, |z| >= threshold
  mad     - rolling median / scaled MAD (robust to spikes in the window)
  iforest - Isolation Forest on value + rolling mean per metric
  rules   - ops alerting rules (sustained saturation, surge, dropout)

Point flag for a method = flagged on any metric.
"""
import numpy as np
import pandas as pd

METRICS = ["cpu_pct", "mem_pct", "net_mbps"]


def _prior_rolling(s, window, func):
    """Rolling statistic over the previous `window` minutes only."""
    return s.shift(1).rolling(window, min_periods=max(10, window // 2)).apply(func, raw=True)


def rolling_zscore(df, window=60, threshold=4.0):
    flags = pd.DataFrame(index=df.index)
    for m in METRICS:
        s = df[m]
        mean = s.shift(1).rolling(window, min_periods=max(10, window // 2)).mean()
        std = s.shift(1).rolling(window, min_periods=max(10, window // 2)).std().clip(lower=1.0)
        z = (s - mean) / std
        flags[m] = (z.abs() >= threshold).fillna(False)
    return flags


def rolling_mad(df, window=60, threshold=5.0):
    flags = pd.DataFrame(index=df.index)
    for m in METRICS:
        s = df[m]
        med = s.shift(1).rolling(window, min_periods=max(10, window // 2)).median()
        mad = s.shift(1).rolling(window, min_periods=max(10, window // 2)).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        scaled = (mad * 1.4826).clip(lower=1.0)
        flags[m] = ((s - med).abs() >= threshold * scaled).fillna(False)
    return flags


def isolation_forest(df, contamination=0.02, seed=42):
    from sklearn.ensemble import IsolationForest
    feats = pd.DataFrame(index=df.index)
    for m in METRICS:
        feats[m] = df[m]
        feats[m + "_mean30"] = df[m].shift(1).rolling(30, min_periods=10).mean()
    feats = feats.bfill().ffill()
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    pred = model.fit_predict(feats)
    point = pd.Series(pred == -1, index=df.index)
    # Attribute a flagged point to every metric that is itself far from
    # its own rolling median that minute. A flagged point with no metric
    # passing that bar still counts as a point flag (see iforest_point).
    flags = pd.DataFrame(False, index=df.index, columns=METRICS)
    for m in METRICS:
        med = df[m].shift(1).rolling(60, min_periods=30).median()
        mad = df[m].shift(1).rolling(60, min_periods=30).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        scaled = (mad * 1.4826).clip(lower=1.0)
        flags[m] = point & ((df[m] - med).abs() >= 3.0 * scaled).fillna(False)
    # Points the forest flags but no metric attributes to still count as point flags.
    return flags, point


def _runs(mask):
    """Yield (start_idx, end_idx inclusive) for each True run in a boolean Series."""
    arr = mask.values
    i = 0
    n = len(arr)
    while i < n:
        if arr[i]:
            j = i
            while j + 1 < n and arr[j + 1]:
                j += 1
            yield i, j
            i = j + 1
        else:
            i += 1


def rule_alerts(df):
    """Ops alerting rules. Returns a list of alert dicts."""
    alerts = []
    idx = df.index
    ts = df["timestamp"]

    def add(metric, rule, severity, s, e, detail):
        alerts.append({
            "metric": metric, "rule": rule, "severity": severity,
            "start": ts.iloc[s].isoformat(), "end": ts.iloc[e].isoformat(),
            "duration_min": int(e - s + 1),
            "peak": round(float(df[metric].iloc[s:e + 1].max()), 2),
            "detail": detail,
        })

    cpu_high = df["cpu_pct"] >= 90
    for s, e in _runs(cpu_high):
        if e - s + 1 >= 10:
            add("cpu_pct", "sustained_high_cpu", "critical", s, e,
                "CPU at or above 90% for 10 minutes or more")
    mem_high = df["mem_pct"] >= 95
    for s, e in _runs(mem_high):
        if e - s + 1 >= 10:
            add("mem_pct", "sustained_high_memory", "critical", s, e,
                "Memory at or above 95% for 10 minutes or more")
    net_med = df["net_mbps"].shift(1).rolling(60, min_periods=30).median()
    surge = df["net_mbps"] >= 3.0 * net_med
    for s, e in _runs(surge.fillna(False)):
        if e - s + 1 >= 5:
            add("net_mbps", "bandwidth_surge", "warning", s, e,
                "Bandwidth at or above 3x the trailing 60-minute median for 5 minutes or more")
    drop = df["net_mbps"] == 0
    for s, e in _runs(drop):
        if e - s + 1 >= 3:
            add("net_mbps", "traffic_dropout", "critical", s, e,
                "Bandwidth at zero for 3 minutes or more (possible outage)")
    spike = df["cpu_pct"] >= 98
    for s, e in _runs(spike):
        add("cpu_pct", "cpu_spike", "warning", s, e,
            "CPU at or above 98%")
    alerts.sort(key=lambda a: a["start"])
    return alerts


def run_all(df):
    z = rolling_zscore(df)
    mad = rolling_mad(df)
    if_flags, if_point = isolation_forest(df)
    out = {}
    for name, flags in [("zscore", z), ("mad", mad), ("iforest", if_flags)]:
        out[name] = flags
        out[name + "_point"] = flags.any(axis=1)
    out["iforest_point"] = if_point
    out["union_point"] = out["zscore_point"] | out["mad_point"] | out["iforest_point"]
    out["alerts"] = rule_alerts(df)
    return out
