"""Offline temperature fitting, model selection and exact matched evaluation."""
import csv
import gzip
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pi.station_features import KEYS, predictors

HOUR = timedelta(hours=1)
H = 96
ALPHAS = (0.001, 0.01, 0.05, 0.15)
BANDS = ((0, 12), (0, 24), (24, 48), (48, 72))


def load_data():
    with (ROOT / 'observations.csv').open() as f:
        by = {datetime.fromisoformat(r['time_lst']): r for r in csv.DictReader(f)}
    origins, inputs, targets = [], [], []
    for now in sorted(by):
        try:
            _, x = predictors(by, now)
        except ValueError:
            continue
        future = [by.get(now + h * HOUR) for h in range(1, H + 1)]
        hourly = np.array([float(r[KEYS[0]]) if r else np.nan for r in future])
        start = 23 - now.hour
        days = hourly[start:start + 72].reshape(3, 24)
        origins.append(now)
        inputs.append(x)
        targets.append(np.r_[hourly, days.min(axis=1), days.max(axis=1)])
    return origins, np.array(inputs), np.array(targets)


def fit_candidates(x, y):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    z = (x - mean) / scale
    intercept = y.mean(axis=0)
    gram, rhs = z.T @ z, z.T @ (y - intercept)
    for penalty in ALPHAS:
        weights = np.linalg.solve(gram + len(x) * penalty * np.eye(len(mean)), rhs)
        yield penalty, (mean, scale, weights, intercept)


def evaluate(x, params):
    mean, scale, weights, intercept = params
    return (x - mean) / scale @ weights + intercept


def main():
    origins, x, y = load_data()
    complete = np.isfinite(y).all(axis=1)
    # Four chronological validation blocks across seasons. Every fit stops
    # before its own validation block, including the 96-hour label horizon.
    blocks = [('2025-10-01','2026-01-01'), ('2026-01-01','2026-04-01'),
              ('2026-04-01','2026-07-01'), ('2026-07-01','2026-10-01')]
    scores = {a: [] for a in ALPHAS}
    for lo, hi in blocks:
        lo, hi = datetime.fromisoformat(lo), datetime.fromisoformat(hi)
        train = complete & [t + H * HOUR < lo for t in origins]
        valid = complete & [lo <= t and t + H * HOUR < hi and t.hour % 6 == 0 for t in origins]
        for a, params in fit_candidates(x[train], y[train]):
            err = abs(evaluate(x[valid], params) - y[valid])
            scores[a].append([float(err[:, :72].mean()), float(err[:, 96:].mean())])
    # Select separate penalties for hourly and daily targets, without test labels.
    hourly_a, daily_a = [min(ALPHAS, key=lambda a: np.mean(scores[a], axis=0)[i]) for i in (0, 1)]
    # Same training cutoff as the uploaded model for a like-for-like comparison.
    end = datetime(2026, 9, 1)
    train = complete & [t + H * HOUR < end for t in origins]
    fits = dict(fit_candidates(x[train], y[train]))
    mean, scale, weights, intercept = fits[hourly_a]
    weights = weights.copy()
    weights[:, 96:] = fits[daily_a][2][:, 96:]
    with gzip.open(ROOT / 'model.json.gz', 'rt') as f:
        old = json.load(f)
    # Keep incumbent outputs for the long horizon and for the daily high.
    legacy_w, legacy_b = old['weights'], old['intercept']
    export_cols = list(range(24)) + list(range(96,99))
    parts = [mean, scale, weights[:, export_cols].T, intercept[export_cols], old['mean'], old['scale'], legacy_w, legacy_b]
    with gzip.GzipFile(str(ROOT / 'pi' / 'weights.bin.gz'), 'wb', mtime=0) as f:
        f.write(b'C484v3\0\0')
        f.write(np.array([len(mean), len(old['mean'])], dtype='<u4').tobytes())
        for part in parts:
            f.write(np.asarray(part, dtype='<f4').tobytes())
    from pi.forecast import load_model, predict, forecast
    model = load_model()
    with (ROOT / 'test_predictions.csv').open() as f:
        matched = {datetime.fromisoformat(r['origin_lst']) for r in csv.DictReader(f)}
    mask = np.array([t in matched for t in origins])
    assert mask.sum() == len(matched) == 152
    times = [t for t in origins if t in matched]
    xx, yy = x[mask], y[mask]
    ref = np.array(old['weights']).T
    old_flat = (xx[:, :51] - old['mean']) / old['scale'] @ ref + old['intercept']
    old_full = old_flat.reshape(-1, 96, 3) + xx[:, None, :3]
    expected = evaluate(xx, (mean, scale, weights, intercept))
    new_h, new_d, old_d = [], [], []
    with (ROOT / 'observations.csv').open() as f:
        by = {datetime.fromisoformat(r['time_lst']): r for r in csv.DictReader(f)}
    for i, now in enumerate(times):
        curve, lows, high_curve = predict(xx[i,:51], xx[i], model)
        blend = np.clip((23 - np.arange(96)) / 12, 0, 1)
        expected_curve = old_full[i,:,0] + blend * (expected[i,:96] - old_full[i,:,0])
        np.testing.assert_allclose(np.array(curve)[:,0], expected_curve, atol=0.0001)
        np.testing.assert_allclose(high_curve, old_full[i,:,0], atol=0.0001)
        np.testing.assert_allclose(np.array(curve)[:,1:], old_full[i,:,1:], atol=0.0001)
        np.testing.assert_allclose(lows, expected[i,96:99], atol=0.0001)
        new_h.append(np.array(curve)[:,0])
        result = forecast([by[now - lag * HOUR] for lag in (0,1,3,6,12,24,48,72)], model, now)
        days = result['next_3_days']
        new_d.append([d[key] for key in ('min_temperature_c','max_temperature_c') for d in days])
        start = 23 - now.hour
        d = old_full[i,start:start+72,0].reshape(3,24)
        old_d.append(np.r_[d.min(axis=1), d.max(axis=1)].round(1))
    def metrics(hourly, daily):
        err = abs(hourly - yy[:,:96])
        return dict(hourly_mae=[float(err[:,lo:hi].mean()) for lo,hi in BANDS],
                    daily_mae=np.nanmean(abs(daily - yy[:,96:]),axis=0).tolist())
    report = dict(training_origins=int(train.sum()), matched_origins=len(times),
                  test_first=str(times[0]),test_last=str(times[-1]), validation_blocks=blocks,
                  validation_mae=scores, hourly_penalty=hourly_a, daily_penalty=daily_a,
                  hourly_bands=BANDS, daily_order='min1,min2,min3,max1,max2,max3',
                  daily_counts=np.isfinite(yy[:,96:]).sum(axis=0).tolist(),
                  uploaded=metrics(old_full[:,:,0], old_d), upgraded=metrics(new_h, new_d))
    (ROOT / 'upgrade' / 'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
