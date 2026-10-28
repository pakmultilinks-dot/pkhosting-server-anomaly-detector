"""Build the static dashboard (dashboard/index.html + dashboard/data/*).

Charts use Chart.js from a CDN with decimation enabled, so the full
1-minute series (20,160 points per metric) renders without the page
stalling. Anomaly markers are the union-method flags. All tables are
rendered from the same files the pipeline writes, so the dashboard
never disagrees with the evaluation.
"""
import json
import os
import shutil

import pandas as pd


def build(df, events, alerts, evaluation, flags, out_dir):
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    df.to_csv(os.path.join(out_dir, "data", "metrics.csv"), index=False)
    events.to_csv(os.path.join(out_dir, "data", "events.csv"), index=False)
    pd.DataFrame(alerts).to_csv(os.path.join(out_dir, "data", "alerts.csv"), index=False)
    with open(os.path.join(out_dir, "data", "evaluation.json"), "w") as f:
        json.dump(evaluation, f, indent=2)

    union = flags["union_point"]
    payload = {
        "timestamps": df["timestamp"].astype(str).tolist(),
        "cpu": df["cpu_pct"].tolist(),
        "mem": df["mem_pct"].tolist(),
        "net": df["net_mbps"].tolist(),
        "anomaly_idx": [int(i) for i in __import__("numpy").where(union.values)[0]],
        "anomaly_cpu": [float(df["cpu_pct"].iloc[i]) for i in __import__("numpy").where(union.values)[0]],
        "anomaly_mem": [float(df["mem_pct"].iloc[i]) for i in __import__("numpy").where(union.values)[0]],
        "anomaly_net": [float(df["net_mbps"].iloc[i]) for i in __import__("numpy").where(union.values)[0]],
    }
    with open(os.path.join(out_dir, "data", "series.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    detail = evaluation["_event_detail"]
    ev = evaluation["union"]
    crit = sum(1 for a in alerts if a["severity"] == "critical")
    # Status in the final hour of the dataset.
    last_hour = df.tail(60)
    tail_alerts = [a for a in alerts
                   if pd.to_datetime(a["end"]) >= pd.to_datetime(df["timestamp"].iloc[-60])]
    status = "OK"
    if any(a["severity"] == "critical" for a in tail_alerts):
        status = "CRITICAL"
    elif tail_alerts:
        status = "WARNING"

    def alert_rows():
        rows = ""
        for a in alerts:
            rows += (f"<tr><td>{a['severity']}</td><td>{a['metric']}</td><td>{a['rule']}</td>"
                     f"<td>{a['start']}</td><td>{a['end']}</td><td>{a['duration_min']}</td>"
                     f"<td>{a['peak']}</td><td>{a['detail']}</td></tr>")
        return rows or "<tr><td colspan='8'>No alerts</td></tr>"

    def event_rows():
        rows = ""
        for d in detail:
            tick = lambda b: "YES" if b else "no"
            rows += (f"<tr><td>{d['event_id']}</td><td>{d['metric']}</td><td>{d['kind']}</td>"
                     f"<td>{d['start']}</td><td>{d['duration_min']}</td><td>{d['peak']}</td>"
                     f"<td>{tick(d['caught_by_zscore'])}</td><td>{tick(d['caught_by_mad'])}</td>"
                     f"<td>{tick(d['caught_by_iforest'])}</td></tr>")
        return rows

    def eval_rows():
        rows = ""
        for name in ["zscore", "mad", "iforest", "union"]:
            e = evaluation[name]
            rows += (f"<tr><td>{name}</td><td>{e['precision']}</td><td>{e['recall']}</td>"
                     f"<td>{e['f1']}</td><td>{e['false_positive_rate']}</td>"
                     f"<td>{e['event_precision']}</td><td>{e['event_recall']}</td>"
                     f"<td>{e['event_f1']}</td><td>{e['median_detection_delay_min']}</td>"
                     f"<td>{e['flagged_minutes']}</td><td>{e['false_alarm_events']}</td></tr>")
        return rows

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PK Hosting Server Anomaly Detector</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
body {{ font-family: system-ui, Arial, sans-serif; margin: 0 auto; max-width: 1100px; padding: 20px; color: #1a1a1a; }}
h1 {{ font-size: 26px; }} h2 {{ margin-top: 34px; }}
.banner {{ padding: 14px; border-radius: 8px; font-weight: bold; background: #e8f5e9; }}
.banner.CRITICAL {{ background: #fdecea; }} .banner.WARNING {{ background: #fff4e0; }}
.cards {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 18px; min-width: 140px; }}
.card b {{ font-size: 26px; display: block; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
th {{ background: #f2f2f2; }}
canvas {{ width: 100%; height: 260px; }}
.note {{ color: #555; font-size: 14px; }}
</style></head><body>
<h1>PK Hosting Server Anomaly Detector</h1>
<p class="note">14 days of per-minute server metrics (CPU %, memory %, bandwidth Mbps) with labelled injected
anomalies. Detectors: rolling z-score, rolling median-MAD, Isolation Forest, plus rule-based alerts.
Status below reflects the final hour of the dataset.</p>
<div class="banner {status}">Status in the final hour: {status}</div>
<div class="cards">
<div class="card"><b>{ev['events_total']}</b> injected events</div>
<div class="card"><b>{ev['events_detected']}</b> events detected (union)</div>
<div class="card"><b>{len(ev['events_missed'])}</b> events missed</div>
<div class="card"><b>{ev['false_alarm_events']}</b> false alarm events</div>
<div class="card"><b>{len(alerts)}</b> rule alerts</div>
<div class="card"><b>{crit}</b> critical alerts</div>
</div>
<h2>CPU usage (%)</h2><canvas id="c_cpu"></canvas>
<h2>Memory usage (%)</h2><canvas id="c_mem"></canvas>
<h2>Bandwidth (Mbps)</h2><canvas id="c_net"></canvas>
<p class="note">Red dots are minutes flagged by any detector (union method). Full data:
<a href="data/metrics.csv">metrics.csv</a>, <a href="data/events.csv">events.csv</a>,
<a href="data/alerts.csv">alerts.csv</a>, <a href="data/evaluation.json">evaluation.json</a>.</p>
<h2>Alerts (rule-based)</h2>
<table><tr><th>Severity</th><th>Metric</th><th>Rule</th><th>Start</th><th>End</th><th>Minutes</th><th>Peak</th><th>Detail</th></tr>
{alert_rows()}</table>
<h2>Injected events and which method caught them</h2>
<table><tr><th>ID</th><th>Metric</th><th>Kind</th><th>Start</th><th>Minutes</th><th>Peak</th><th>z-score</th><th>MAD</th><th>Isolation Forest</th></tr>
{event_rows()}</table>
<h2>Evaluation (ground truth = injected events)</h2>
<table><tr><th>Method</th><th>Point precision</th><th>Point recall</th><th>Point F1</th><th>Point FPR</th><th>Event precision</th><th>Event recall</th><th>Event F1</th><th>Median delay (min)</th><th>Flagged minutes</th><th>False alarm events</th></tr>
{eval_rows()}</table>
<h2>How this works</h2>
<p>Simulated data is the primary dataset because PK Hosting production servers are not accessible to this
project (no dashboard login, SSH or API access was ever connected). The generator builds realistic diurnal and
weekly patterns with noise, then injects labelled anomalies (spikes, level shifts, saturation, leaks, bandwidth
surges, dropouts). Every detector uses only past data, so a flag never uses information from after its minute.
Event-level scoring counts an event as detected if any flag falls within the event window plus or minus 5 minutes.
To run on a real server, use <code>src/collect_real.py</code> (reads /proc and /sys, standard library only) and
feed its CSV through the same detectors. Full method notes are in the project README.</p>
<script>
fetch('data/series.json').then(r => r.json()).then(d => {{
  const labels = d.timestamps.map(t => t.slice(5, 16));
  function chart(id, values, anom, unit) {{
    const pts = d.anomaly_idx.map((idx, k) => ({{ x: idx, y: anom[k] }}));
    new Chart(document.getElementById(id), {{ type: 'line',
      data: {{ labels: labels, datasets: [
        {{ label: unit, data: values, borderColor: '#2563eb', borderWidth: 1, pointRadius: 0 }},
        {{ label: 'flagged', data: pts, type: 'scatter', backgroundColor: '#dc2626', pointRadius: 3, showLine: false }}
      ]}},
      options: {{ responsive: true, animation: false, parsing: false,
        plugins: {{ decimation: {{ enabled: true, algorithm: 'min-max' }} }},
        scales: {{ x: {{ ticks: {{ maxTicksLimit: 14 }} }} }} }}
    }});
  }}
  chart('c_cpu', d.cpu, d.anomaly_cpu, 'CPU %');
  chart('c_mem', d.mem, d.anomaly_mem, 'Memory %');
  chart('c_net', d.net, d.anomaly_net, 'Mbps');
}});
</script></body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)
    return out_dir
