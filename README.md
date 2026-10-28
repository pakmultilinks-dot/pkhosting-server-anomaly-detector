# PK Hosting Server Anomaly Detector

Flags unusual behaviour in server metrics (CPU %, memory %, bandwidth Mbps)
for PK Hosting, with a dashboard and alerting view showing every detected
anomaly. Built for the PKHosting.com project task (10 points, due 14 Oct 2026).

Live dashboard: see the deployed link in the submission report.
Source: this repo. Run it yourself: `python3 run.py`, then open
`dashboard/index.html` (or serve the `dashboard/` folder).

## Data source, and why

**Primary dataset: simulated, with a clear justification.** PK Hosting
production servers are not accessible to this project: no dashboard login,
SSH key or API access was ever connected, so real production metrics cannot
be pulled. Inventing a claim of real production data would be dishonest, so
the evaluated dataset is simulated instead, built to behave like a real
hosting server:

- 14 days of per-minute data (20,160 rows), 1 Oct 2026 onwards.
- Realistic patterns: a diurnal cycle (busy afternoons, quiet nights), a
  weekday/weekend effect, and Gaussian noise.
- 24 labelled injected anomalies (1,393 anomalous minutes) of the kinds an
  ops team actually sees: point spikes, level shifts, sustained saturation,
  slow memory leaks, bandwidth surges and traffic dropouts.

**Real-data path included.** `src/collect_real.py` samples a real Linux
server from `/proc` and `/sys` (CPU from `/proc/stat`, memory from
`/proc/meminfo`, throughput from interface byte counters) using only the
standard library, and writes rows in exactly the simulated schema. Point the
detectors at its CSV and the same code runs unchanged on real data. That is
the honest shape of "real, simulated or both": simulated for evaluation,
real-ready by construction.

## Detection methods

Every detector uses only past data (rolling windows are shifted by one
minute), so a flag at minute t never uses information from after t, which
is how the system would run live.

| Method | How it works | Settings |
|---|---|---|
| Rolling z-score | Value vs trailing mean/std | 60-min window, \|z\| >= 4 |
| Rolling median-MAD | Value vs trailing median / scaled MAD (robust to spikes inside the window) | 60-min window, >= 5 MADs |
| Isolation Forest | Unsupervised tree ensemble on value + 30-min rolling mean per metric | contamination 0.02, 200 trees |
| Rule alerts | Ops rules, not statistics | CPU >= 90% for 10+ min (critical); memory >= 95% for 10+ min (critical); bandwidth >= 3x trailing median for 5+ min (warning); bandwidth 0 for 3+ min (critical, possible outage); CPU >= 98% (warning) |

The dashboard's markers use the **union** of the three statistical methods
(flagged by any of them).

## Evaluation method (documented)

Ground truth is the injected event list, so evaluation is exact, not judged
by eye. Two levels are reported because an ops team cares about both:

- **Point level:** every minute classified right or wrong (precision,
  recall, F1, false positive rate).
- **Event level:** an injected event counts as detected if any flagged
  minute falls inside the event window expanded by 5 minutes on each side
  (a flag a few minutes early or late still catches the incident).
  Detection delay is minutes from event start to the first flag. A flagged
  run overlapping no true event is a false alarm event.

## Results (14-day simulated set, seed 42)

| Method | Point P / R / F1 | Event P / R / F1 | Events detected | Median delay | False alarm events |
|---|---|---|---|---|---|
| z-score | 0.64 / 0.06 / 0.10 | 0.59 / 1.00 / 0.74 | 24 / 24 | 0 min | 17 |
| median-MAD | 0.55 / 0.22 / 0.31 | 0.69 / 1.00 / 0.81 | 24 / 24 | 0 min | 11 |
| Isolation Forest | 1.00 / 0.29 / 0.45 | 0.92 / 0.50 / 0.65 | 12 / 24 | 11 min | 1 |
| Union | 0.70 / 0.44 / 0.54 | 0.55 / 1.00 / 0.71 | 24 / 24 | 0 min | 20 |

Rule alerts fired: 15 (7 critical), covering the sustained saturations,
surges and all three traffic dropouts.

What the numbers mean, honestly:

- **Median-MAD is the best single method** (event F1 0.81). It catches every
  event and has the fewest false alarms of the rolling methods.
- **Rolling methods are onset detectors, not duration trackers.** Once an
  anomaly enters the trailing window, the baseline adapts and later minutes
  of a long event stop flagging. That is why point recall is low while
  event recall is 1.00: every event is caught, but not every minute of it.
- **Isolation Forest is precise but narrow.** Point precision 0.995 and
  only 1 false alarm event, but it misses brief 2-3 minute spikes and all
  three dropouts: a short spike from a night trough is not globally extreme
  enough to rank in the top 2% of anomalies. Raising contamination to 0.05
  lifts event recall to 0.79 but drops event F1 to 0.48 (37 false alarms),
  so 0.02 stays.
- **The union catches everything (24/24) at the cost of 20 false alarm
  events**, mostly around steep morning ramps where normal behaviour looks
  like a shift. For an alerting view, the rule layer is the triage: it fires
  on sustained, operationally meaningful conditions only.

## Dashboard and alerting view

`dashboard/index.html` (generated by `run.py`) shows: a status banner for
the final hour of the dataset, summary cards, one chart per metric with
anomaly markers, the rule-alert table, the injected-event table with which
method caught each event, and the full evaluation table. Raw data is linked
from the dashboard: `data/metrics.csv`, `data/events.csv`,
`data/alerts.csv`, `data/evaluation.json`, `data/series.json`.

## How to run

```bash
pip install -r requirements.txt
python3 run.py
python3 tests/test_detectors.py
```

To collect real data on a Linux server (no dependencies):

```bash
python3 src/collect_real.py --seconds 300 --interval 5 --out data/real.csv
```

## Project structure

```
run.py                  pipeline: generate -> detect -> alert -> evaluate -> dashboard
src/generate.py         simulated metrics + injected events (ground truth)
src/detect.py           z-score, median-MAD, Isolation Forest, rule alerts
src/evaluate.py         point-level and event-level evaluation
src/collect_real.py     /proc + /sys sampler, same schema as simulated data
src/build_dashboard.py  static dashboard generator
tests/test_detectors.py plain-assert tests (no framework needed)
data/                   generated metrics, events, evaluation
dashboard/              deployable static site (index.html + data/)
```

## Limitations

- The evaluated numbers come from simulated data. Real server behaviour
  (noisy neighbours, backup windows, deployments) will differ, and the
  thresholds above should be re-tuned on real data before paging anyone.
- The dashboard is a static snapshot of the latest run, not a streaming
  service. Re-run `run.py` (or schedule it) to refresh it.
- False alarms cluster on steep normal ramps. A production version should
  add time-of-day baselines before using the union method for paging.
