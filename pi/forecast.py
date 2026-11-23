"""Offline CART 484 inference. Python standard library only; fixed UTC-5."""
import csv
import gzip
import json
import struct
import sys
from array import array
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__:
    from .station_features import predictors
else:
    from station_features import predictors

ROOT = Path(__file__).resolve().parent
ONE_HOUR = timedelta(hours=1)
HISTORY = timedelta(hours=72)


def load_model(path=None):
    with gzip.open(path or ROOT / 'weights.bin.gz', 'rb') as stream:
        if stream.read(8) != b'C484v3\0\0':
            raise ValueError('Invalid CART 484 model header')
        features, legacy = struct.unpack('<II', stream.read(8))
        if (features, legacy) != (124, 51):
            raise ValueError('Incompatible feature dimensions')
        sizes = [features, features, 27 * features, 27, legacy, legacy, 288 * legacy, 288]
        parts = []
        for size in sizes:
            values = array('f')
            values.frombytes(stream.read(4 * size))
            if len(values) != size:
                raise ValueError('Truncated model')
            if sys.byteorder != 'little':
                values.byteswap()
            parts.append(values)
        if stream.read(1):
            raise ValueError('Unexpected trailing model data')
    return parts


def linear(x, mean, scale, weights, bias):
    z = [(v - m) / s for v, m, s in zip(x, mean, scale)]
    n = len(mean)
    view = memoryview(weights)
    return [b + sum(a * v for a, v in zip(view[i*n:(i+1)*n], z)) for i, b in enumerate(bias)]


def predict(base, features, model):
    mean, scale, weights, bias, old_mean, old_scale, old_weights, old_bias = model
    temps = linear(features, mean, scale, weights, bias)
    offsets = linear(base, old_mean, old_scale, old_weights, old_bias)
    legacy = [[offsets[3*h+k] + base[k] for k in range(3)] for h in range(96)]
    highs_curve = [r[0] for r in legacy]
    for h in range(24):
        # Use the short-range model through hour 12, then fade to the
        # incumbent by hour 24 to avoid a jump at the handover.
        weight = min(1., (23 - h) / 12)
        legacy[h][0] += weight * (temps[h] - legacy[h][0])
    return legacy, temps[24:], highs_curve


def forecast(rows, model, reference_time=None):
    history = {}
    now = None
    for row in rows:
        stamp = datetime.fromisoformat(row['time_lst'])
        if stamp.tzinfo is not None or stamp.minute or stamp.second or stamp.microsecond:
            raise ValueError('Expected hourly timestamps in fixed local standard time (UTC-5)')
        now = max(now, stamp) if now else stamp
        if stamp >= now - HISTORY:
            history[stamp] = row
        if len(history) > 145:
            history = {t: r for t, r in history.items() if t >= now - HISTORY}
    if now is None:
        raise ValueError('No observations')
    if reference_time is not None and not 0 <= (reference_time - now).total_seconds() <= 5400:
        raise ValueError('Latest hourly input is in the future or over 90 minutes old')
    base, features = predictors(history, now)
    numbers, lows, highs_curve = predict(base, features, model)
    hourly = []
    for h, (temp, rh, p) in enumerate(numbers, 1):
        rh = max(0., min(100., rh))
        sky = ('Risque de précipitations' if rh >= 85 and p - base[2] <= -0.3 else
               'Nuageux' if rh >= 75 else 'Partiellement nuageux' if rh >= 55 else 'Possiblement dégagé')
        hourly.append(dict(time_lst=str(now + h * ONE_HOUR), temperature_c=round(temp,1),
                           humidity_pct=round(rh,1), pressure_kpa=round(p,2), sky_estimate=sky))
    daily = []
    for d in range(3):
        start = 23 - now.hour + 24*d
        day = hourly[start:start+24]
        labels = [r['sky_estimate'] for r in day]
        high = max(highs_curve[start:start+24])
        low = min(lows[d], high)
        # If tomorrow starts inside the displayed 12 hours, include those
        # public hourly estimates in the daily bounds.
        visible = hourly[start:12]
        if visible:
            low = min(low, min(r['temperature_c'] for r in visible))
            high = max(high, max(r['temperature_c'] for r in visible))
        daily.append(dict(date=day[0]['time_lst'][:10], min_temperature_c=round(low,1),
                          max_temperature_c=round(high,1), hours=24,
                          sky_estimate=max(dict.fromkeys(labels), key=labels.count)))
    return dict(issued_at_lst=str(now), experimental=True,
                sky_method='humidity/pressure heuristic; not calibrated probability',
                next_12_hours=hourly[:12], next_3_days=daily)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs='?', default=str(ROOT / 'example_input.csv'))
    parser.add_argument('--live', action='store_true', help='Reject stale or future hourly data')
    args = parser.parse_args()
    reference = datetime.now(timezone(-5 * ONE_HOUR)).replace(tzinfo=None) if args.live else None
    with open(args.csv, encoding='utf-8-sig', newline='') as f:
        print(json.dumps(forecast(csv.DictReader(f), load_model(), reference), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
