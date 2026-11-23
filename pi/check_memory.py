"""Run on Linux with Python 3 to measure the complete standalone runtime."""
import csv
import json
import platform
import resource
import statistics
import time
from pathlib import Path

root = Path(__file__).resolve().parent
from forecast import forecast, load_model

with (root / 'example_input.csv').open() as f:
    rows = list(csv.DictReader(f))
model = load_model()
durations = []
for iteration in range(101):
    start = time.perf_counter()
    result = forecast(rows, model)
    if iteration:
        durations.append(1000 * (time.perf_counter() - start))
print(json.dumps(dict(machine=platform.machine(), python=platform.python_version(),
                      median_ms=statistics.median(durations),
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      model_bytes=(root / 'weights.bin.gz').stat().st_size), indent=2))
