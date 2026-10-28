"""Real-data collector (Linux, standard library only).

Samples this machine's CPU, memory and network throughput from /proc and
/sys, writing rows in exactly the simulated schema
(timestamp, cpu_pct, mem_pct, net_mbps, is_anomaly) with is_anomaly left
blank for the detector to fill in. Run it on a real server and the
detectors in detect.py work unchanged, which is what makes the system
real-data-ready even though the evaluated dataset is simulated.

Usage:
    python3 src/collect_real.py --seconds 60 --interval 5 --out data/real.csv
"""
import argparse
import csv
import time
from datetime import datetime, timezone


def read_cpu_times():
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = [float(x) for x in parts]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals), idle


def read_mem_pct():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k.strip()] = float(v.split()[0])
    return (info["MemTotal"] - info["MemAvailable"]) / info["MemTotal"] * 100.0


def pick_interface():
    import os
    base = "/sys/class/net"
    for iface in sorted(os.listdir(base)):
        if iface == "lo":
            continue
        try:
            with open(os.path.join(base, iface, "operstate")) as f:
                if f.read().strip() == "up":
                    return iface
        except OSError:
            continue
    return "eth0"


def read_net_bytes(iface):
    rx = tx = 0
    for name, acc in (("rx_bytes", "rx"), ("tx_bytes", "tx")):
        try:
            with open(f"/sys/class/net/{iface}/statistics/{name}") as f:
                val = float(f.read().strip())
        except OSError:
            val = 0.0
        if acc == "rx":
            rx = val
        else:
            tx = val
    return rx + tx


def sample(iface, prev_cpu, prev_net, dt):
    total, idle = read_cpu_times()
    net = read_net_bytes(iface)
    cpu_pct = 0.0
    if prev_cpu is not None and dt > 0:
        d_total = total - prev_cpu[0]
        d_idle = idle - prev_cpu[1]
        if d_total > 0:
            cpu_pct = max(0.0, min(100.0, (1 - d_idle / d_total) * 100.0))
    mbps = 0.0
    if prev_net is not None and dt > 0:
        mbps = max(0.0, (net - prev_net) * 8 / dt / 1e6)
    return cpu_pct, read_mem_pct(), mbps, (total, idle), net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--interval", type=float, default=5)
    ap.add_argument("--out", default="data/real.csv")
    args = ap.parse_args()
    iface = pick_interface()
    print(f"Collecting from interface {iface} for {args.seconds}s ...")
    prev_cpu = prev_net = None
    start = time.time()
    rows = []
    while time.time() - start < args.seconds:
        t0 = time.time()
        cpu, mem, mbps, prev_cpu, prev_net = sample(iface, prev_cpu, prev_net, args.interval)
        rows.append([datetime.now(timezone.utc).isoformat(), round(cpu, 2),
                     round(mem, 2), round(mbps, 2), ""])
        time.sleep(max(0, args.interval - (time.time() - t0)))
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "cpu_pct", "mem_pct", "net_mbps", "is_anomaly"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
