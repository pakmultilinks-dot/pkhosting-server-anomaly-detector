"""Pipeline: generate -> detect -> alert -> evaluate -> dashboard.

Usage:
    python3 run.py [--days 14] [--seed 42] [--start 2026-10-01]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate import generate
from detect import run_all
from evaluate import evaluate
from build_dashboard import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2026-10-01")
    args = ap.parse_args()

    df, events = generate(days=args.days, seed=args.seed, start=args.start)
    results = run_all(df)
    ev = evaluate(df, events, results)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/metrics.csv", index=False)
    events.to_csv("data/events.csv", index=False)
    with open("data/evaluation.json", "w") as f:
        json.dump(ev, f, indent=2)
    detail = ev.pop("_event_detail")
    ev_for_dash = dict(ev)
    ev_for_dash["_event_detail"] = detail
    build(df, events, results["alerts"], ev_for_dash, results, "dashboard")

    print(f"rows: {len(df)}, anomalous minutes: {int(df['is_anomaly'].sum())}, events: {len(events)}")
    print(f"rule alerts: {len(results['alerts'])} "
          f"({sum(1 for a in results['alerts'] if a['severity'] == 'critical')} critical)")
    for name in ["zscore", "mad", "iforest", "union"]:
        e = ev[name]
        print(f"{name}: point P={e['precision']} R={e['recall']} F1={e['f1']} | "
              f"event P={e['event_precision']} R={e['event_recall']} F1={e['event_f1']} | "
              f"detected {e['events_detected']}/{e['events_total']} | "
              f"median delay {e['median_detection_delay_min']} min | "
              f"false alarms {e['false_alarm_events']}")
    best = max(["zscore", "mad", "iforest"], key=lambda n: ev[n]["event_f1"])
    print(f"best single method by event F1: {best}")
    print("dashboard: dashboard/index.html")


if __name__ == "__main__":
    main()
