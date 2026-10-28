"""Evaluation against injected ground truth.

Two levels, because an ops team cares about both:
  point level - every minute classified right or wrong.
  event level - an injected event counts as detected if any flagged point
    falls inside the event window expanded by 5 minutes on each side
    (a flag a few minutes early or late still catches the incident).
    Detection delay = minutes from event start to the first flag inside
    the window (0 if the flag starts before the event, which the tolerance
    allows; delays are reported as observed, never negative-clipped in
    the raw output, but summarised with the median).

A flagged run that overlaps no true event (with tolerance) is a false
alarm event. Precision/recall/F1 are reported at both levels.
"""
import numpy as np
import pandas as pd

TOL_MIN = 5


def _point_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "false_positive_rate": round(fpr, 4)}


def _runs(mask):
    arr = np.asarray(mask)
    i, n = 0, len(arr)
    while i < n:
        if arr[i]:
            j = i
            while j + 1 < n and arr[j + 1]:
                j += 1
            yield i, j
            i = j + 1
        else:
            i += 1


def _event_metrics(df, events, point_flag):
    ts = pd.to_datetime(df["timestamp"])
    flag_idx = np.where(np.asarray(point_flag))[0]
    detected = []
    missed = []
    delays = []
    for _, ev in events.iterrows():
        start = pd.to_datetime(ev["start"])
        end = pd.to_datetime(ev["end"])
        lo = start - pd.Timedelta(minutes=TOL_MIN)
        hi = end + pd.Timedelta(minutes=TOL_MIN)
        hits = [i for i in flag_idx if lo <= ts.iloc[i] <= hi]
        if hits:
            detected.append(ev["event_id"])
            delays.append(int((ts.iloc[min(hits)] - start).total_seconds() // 60))
        else:
            missed.append(ev["event_id"])
    # False alarm events: flagged runs overlapping no true event window.
    windows = [(pd.to_datetime(r["start"]) - pd.Timedelta(minutes=TOL_MIN),
                pd.to_datetime(r["end"]) + pd.Timedelta(minutes=TOL_MIN))
               for _, r in events.iterrows()]
    false_alarms = 0
    for s, e in _runs(point_flag):
        t0, t1 = ts.iloc[s], ts.iloc[e]
        if not any(t0 <= hi and t1 >= lo for lo, hi in windows):
            false_alarms += 1
    n_ev = len(events)
    rec = len(detected) / n_ev if n_ev else 0.0
    prec = len(detected) / (len(detected) + false_alarms) if (len(detected) + false_alarms) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "events_total": n_ev, "events_detected": len(detected),
        "events_missed": missed, "false_alarm_events": false_alarms,
        "event_precision": round(prec, 4), "event_recall": round(rec, 4),
        "event_f1": round(f1, 4),
        "median_detection_delay_min": int(np.median(delays)) if delays else None,
        "detection_delays_min": delays,
    }


def evaluate(df, events, results):
    y = df["is_anomaly"].values
    out = {}
    for name in ["zscore", "mad", "iforest", "union"]:
        point = results[name + "_point"].values
        entry = _point_metrics(y, point)
        entry.update(_event_metrics(df, events, point))
        entry["flagged_minutes"] = int(point.sum())
        out[name] = entry
    # Per-event detail: which methods caught each event.
    ts = pd.to_datetime(df["timestamp"])
    detail = []
    for _, ev in events.iterrows():
        start = pd.to_datetime(ev["start"])
        end = pd.to_datetime(ev["end"])
        lo = start - pd.Timedelta(minutes=TOL_MIN)
        hi = end + pd.Timedelta(minutes=TOL_MIN)
        row = {"event_id": ev["event_id"], "metric": ev["metric"], "kind": ev["kind"],
               "start": ev["start"], "end": ev["end"], "duration_min": int(ev["duration_min"]),
               "peak": float(ev["peak"])}
        for name in ["zscore", "mad", "iforest"]:
            flag_idx = np.where(results[name + "_point"].values)[0]
            row["caught_by_" + name] = bool(any(lo <= ts.iloc[i] <= hi for i in flag_idx))
        row["caught_by_any"] = row["caught_by_zscore"] or row["caught_by_mad"] or row["caught_by_iforest"]
        detail.append(row)
    out["_event_detail"] = detail
    return out
