"""Plain-assert tests (no test framework dependency). Run:
    python3 tests/test_detectors.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from generate import generate, EVENTS
from detect import rolling_zscore, rolling_mad, isolation_forest, rule_alerts
from evaluate import evaluate, _point_metrics


def test_generator():
    df, ev = generate()
    assert len(df) == 14 * 1440, len(df)
    assert len(ev) == len(EVENTS)
    assert int(df["is_anomaly"].sum()) == int(ev["duration_min"].sum()), \
        "labels must match total injected minutes (no overlapping events)"
    assert set(df.columns) == {"timestamp", "cpu_pct", "mem_pct", "net_mbps", "is_anomaly"}
    print("generator OK:", len(df), "rows,", len(ev), "events")


def _flag_in_window(point_flag, ts, start, end, tol=5):
    lo = pd.to_datetime(start) - pd.Timedelta(minutes=tol)
    hi = pd.to_datetime(end) + pd.Timedelta(minutes=tol)
    return bool(((ts >= lo) & (ts <= hi) & point_flag.values).any())


def test_detectors_catch_known_events():
    df, ev = generate()
    ts = pd.to_datetime(df["timestamp"])
    z = rolling_zscore(df).any(axis=1)
    mad = rolling_mad(df).any(axis=1)
    _, if_point = isolation_forest(df)
    e01 = ev[ev["event_id"] == "E01"].iloc[0]
    assert _flag_in_window(z, ts, e01["start"], e01["end"]), "z-score missed the E01 CPU spike"
    e04 = ev[ev["event_id"] == "E04"].iloc[0]
    assert _flag_in_window(mad, ts, e04["start"], e04["end"]), "MAD missed the E04 saturation"
    # Isolation Forest is a high-precision, low-recall method: it catches
    # sustained extremes (E04) but misses brief 2-3 minute spikes (E01),
    # which is documented in the evaluation, not hidden by the test.
    assert _flag_in_window(if_point, ts, e04["start"], e04["end"]), "Isolation Forest missed E04"
    union = z | mad | if_point
    e05 = ev[ev["event_id"] == "E05"].iloc[0]
    for eid, e in [("E01", e01), ("E04", e04), ("E05", e05)]:
        assert _flag_in_window(union, ts, e["start"], e["end"]), f"union missed {eid}"
    print("detectors catch known events OK")


def test_rules():
    n = 30
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-01", periods=n, freq="min"),
        "cpu_pct": [97.0] * 15 + [40.0] * 15,
        "mem_pct": [50.0] * n,
        "net_mbps": [50.0] * n,
        "is_anomaly": [0] * n,
    })
    alerts = rule_alerts(df)
    assert any(a["rule"] == "sustained_high_cpu" and a["severity"] == "critical" for a in alerts), alerts
    df2 = df.copy()
    df2["net_mbps"] = [0.0] * 5 + [50.0] * 25
    alerts2 = rule_alerts(df2)
    assert any(a["rule"] == "traffic_dropout" for a in alerts2), alerts2
    print("rule alerts OK")


def test_evaluation_bounds():
    df, ev = generate()
    from detect import run_all
    results = run_all(df)
    out = evaluate(df, ev, results)
    for name in ["zscore", "mad", "iforest", "union"]:
        e = out[name]
        for k in ["precision", "recall", "f1", "false_positive_rate",
                  "event_precision", "event_recall", "event_f1"]:
            assert 0.0 <= e[k] <= 1.0, (name, k, e[k])
    assert out["union"]["events_detected"] >= 18, out["union"]
    m = _point_metrics(df["is_anomaly"].values, df["is_anomaly"].values)
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    print("evaluation bounds OK; union detected",
          out["union"]["events_detected"], "/", out["union"]["events_total"], "events")


def test_real_collector_helpers():
    from collect_real import read_cpu_times, read_mem_pct, pick_interface
    total, idle = read_cpu_times()
    assert total > 0 and 0 <= idle <= total
    mem = read_mem_pct()
    assert 0 < mem < 100
    assert isinstance(pick_interface(), str)
    print("real collector helpers OK")


if __name__ == "__main__":
    test_generator()
    test_detectors_catch_known_events()
    test_rules()
    test_evaluation_bounds()
    test_real_collector_helpers()
    print("ALL TESTS PASSED")
