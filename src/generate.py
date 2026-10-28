"""Simulated server metrics with labelled, injected anomalies.

Why simulated data is the primary dataset (see README for the full
justification): PK Hosting production servers are not accessible to this
project (no dashboard login, SSH or API access was ever connected), so
real production metrics cannot be pulled. The generator below produces
realistic series (diurnal + weekly patterns, noise) and injects labelled
anomalies of the kinds an ops team actually sees, which makes detection
measurable: every injected event is ground truth for evaluation.

Schema (one row per minute):
    timestamp, cpu_pct, mem_pct, net_mbps, is_anomaly
plus events.csv: one row per injected event.
"""
import numpy as np
import pandas as pd

METRICS = ["cpu_pct", "mem_pct", "net_mbps"]

# (event_id, metric, kind, start_day, start_hour, duration_min, magnitude)
# magnitude meaning depends on kind: spike/surge = additive peak,
# shift/saturation = level the metric is pushed to, leak = total rise,
# dropout = forced value (0).
EVENTS = [
    ("E01", "cpu_pct", "spike", 0, 10, 3, 55),
    ("E02", "net_mbps", "surge", 1, 14, 15, 9.0),
    ("E03", "mem_pct", "leak", 2, 2, 360, 22),
    ("E04", "cpu_pct", "saturation", 3, 11, 45, 97),
    ("E05", "net_mbps", "dropout", 4, 3, 12, 0),
    ("E06", "cpu_pct", "spike", 5, 19, 2, 48),
    ("E07", "mem_pct", "shift", 6, 9, 120, 88),
    ("E08", "net_mbps", "surge", 6, 21, 20, 7.0),
    ("E09", "cpu_pct", "saturation", 7, 13, 30, 96),
    ("E10", "cpu_pct", "spike", 8, 8, 4, 60),
    ("E11", "mem_pct", "leak", 8, 20, 300, 18),
    ("E12", "net_mbps", "dropout", 9, 4, 8, 0),
    ("E13", "cpu_pct", "shift", 10, 15, 90, 82),
    ("E14", "net_mbps", "surge", 11, 12, 10, 10.0),
    ("E15", "mem_pct", "spike", 11, 23, 3, 35),
    ("E16", "cpu_pct", "saturation", 12, 10, 60, 98),
    ("E17", "net_mbps", "dropout", 12, 2, 20, 0),
    ("E18", "cpu_pct", "spike", 13, 17, 2, 52),
    ("E19", "mem_pct", "shift", 13, 6, 150, 85),
    ("E20", "net_mbps", "surge", 13, 20, 25, 8.0),
    ("E21", "cpu_pct", "shift", 1, 22, 60, 78),
    ("E22", "mem_pct", "spike", 4, 16, 2, 30),
    ("E23", "net_mbps", "surge", 9, 18, 12, 6.0),
    ("E24", "cpu_pct", "saturation", 10, 2, 40, 95),
]


def _base_series(n, rng, day_phase, weekly_amp, base, amp, noise):
    """Diurnal sine + weekly effect + Gaussian noise, clipped to range."""
    t = np.arange(n)
    hour = (t % 1440) / 60.0
    day = (t // 1440) % 7
    weekend = np.isin(day, [5, 6]).astype(float)
    series = (
        base
        + amp * np.sin(2 * np.pi * (hour - day_phase) / 24.0)
        - weekly_amp * weekend
        + rng.normal(0, noise, n)
    )
    return series


def generate(days=14, seed=42, start="2026-10-01"):
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range(start=start, periods=n, freq="min")
    cpu = _base_series(n, rng, day_phase=14, weekly_amp=6, base=42, amp=16, noise=4)
    mem = _base_series(n, rng, day_phase=10, weekly_amp=2, base=58, amp=5, noise=2)
    net = _base_series(n, rng, day_phase=15, weekly_amp=15, base=55, amp=30, noise=8)
    cpu = np.clip(cpu, 2, 99)
    mem = np.clip(mem, 20, 98)
    net = np.clip(net, 1, 900)

    labels = np.zeros(n, dtype=int)
    events = []
    for eid, metric, kind, day, hour, dur, mag in EVENTS:
        start_i = day * 1440 + hour * 60
        end_i = min(start_i + dur, n)
        if start_i >= n:
            continue
        series = {"cpu_pct": cpu, "mem_pct": mem, "net_mbps": net}[metric]
        seg = series[start_i:end_i]
        if kind == "spike":
            series[start_i:end_i] = np.clip(seg + mag, 0, 99 if metric != "net_mbps" else 1200)
        elif kind in ("shift", "saturation"):
            series[start_i:end_i] = mag
        elif kind == "leak":
            ramp = np.linspace(0, mag, end_i - start_i)
            series[start_i:end_i] = np.clip(seg + ramp, 0, 98)
        elif kind == "surge":
            series[start_i:end_i] = seg * mag
        elif kind == "dropout":
            series[start_i:end_i] = 0
        labels[start_i:end_i] = 1
        events.append({
            "event_id": eid, "metric": metric, "kind": kind,
            "start": idx[start_i].isoformat(), "end": idx[end_i - 1].isoformat(),
            "duration_min": end_i - start_i,
            "peak": round(float(np.max(series[start_i:end_i])), 2),
        })

    df = pd.DataFrame({
        "timestamp": idx, "cpu_pct": np.round(cpu, 2),
        "mem_pct": np.round(mem, 2), "net_mbps": np.round(net, 2),
        "is_anomaly": labels,
    })
    return df, pd.DataFrame(events)


if __name__ == "__main__":
    df, ev = generate()
    print(df.head())
    print("rows:", len(df), "anomalous minutes:", int(df["is_anomaly"].sum()))
    print("events:", len(ev))
