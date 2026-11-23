"""Features computed only from the eight sensor samples available at issue time."""
import math
from datetime import datetime, timedelta

KEYS = ('temperature_c', 'humidity_pct', 'pressure_kpa')
LAGS = (0, 1, 3, 6, 12, 24, 48, 72)


def predictors(history, now):
    samples = []
    for lag in LAGS:
        row = history.get(now - timedelta(hours=lag))
        if row is None:
            raise ValueError(f'Missing hourly input at lag {lag} hours')
        values = [float(row[key]) for key in KEYS]
        if not all(map(math.isfinite, values)):
            raise ValueError('Non-finite sensor reading')
        samples.append(values)
    annual = math.tau * (now - datetime(now.year, 1, 1)).total_seconds() / (86400 * 365.2425)
    clock = math.tau * now.hour / 24
    sh, ch = math.sin(clock), math.cos(clock)
    sy, cy = math.sin(annual), math.cos(annual)
    phase = [sh, ch, sy, cy, math.sin(2 * annual), math.cos(2 * annual)]
    trend = [[v - past for v, past in zip(samples[0], values)] for values in samples[1:]]
    base = [v for row in samples + trend for v in row] + phase
    return base, expand(base)


def expand(base):
    """Expand the original 51-feature vector without extra sensor inputs."""
    base = list(base)
    if len(base) != 51:
        raise ValueError('Expected 51 base features')
    t, rh, p = base[:3]
    sh, ch, sy, cy = base[45:49]
    clock, annual = math.atan2(sh, ch), math.atan2(sy, cy)
    # Centering constants only set the scale of physical interaction terms.
    anomaly = (t - (7 - 14 * cy - 4 * sy)) / 10
    humidity = (rh - 70) / 25
    pressure = p - 100.5
    dt3, dh3, dp3 = base[27:30]
    dt12, dh12, dp12 = base[33:36]
    dt24, dh24, dp24 = base[36:39]
    extra = [math.sin(2 * clock), math.cos(2 * clock),
             math.sin(3 * annual), math.cos(3 * annual)]
    factors = [1., anomaly, humidity, pressure, dt3 / 5, dp3,
               dt12 / 8, dh12 / 30, dp12, dt24 / 8, dp24]
    for factor in factors:
        extra.extend(factor * z for z in (sh, ch, sy, cy))
    for a in (sh, ch):
        extra.extend(a * b for b in (sy, cy, math.sin(2 * clock), math.cos(2 * clock)))
    for val in (anomaly, humidity, pressure, dp3, dp12, dp24, dt24 / 8):
        extra.append(abs(val))
    extra.extend([anomaly * pressure, anomaly * dp12, anomaly * humidity,
                  dp12 * humidity, pressure * humidity, pressure * dp12,
                  dt12 * dp12, dt24 * dp24, dh24 * dp24, dh3 * dp3])
    return base + extra
