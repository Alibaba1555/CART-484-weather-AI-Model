"""Earlier-origin retrospective check, refitting both recipes before 2025.
No observations from the scoring period enter either fitted set of coefficients.
Both methods forecast from the same six-hourly origins with full labels.
"""
import json
from datetime import datetime
import numpy as np
from train_upgrade import ROOT, H, HOUR, BANDS, load_data, fit_candidates, evaluate


def main():
    origins, x, y = load_data()
    selection = json.loads((ROOT / 'upgrade' / 'evaluation.json').read_text())
    complete = np.isfinite(y).all(axis=1)
    cutoff = datetime(2025, 1, 1)
    train = complete & [t + H * HOUR < cutoff for t in origins]
    six = np.array([t.hour % 6 == 0 for t in origins])
    # Reproduce the baseline's requirement for the complete previous day.
    import csv
    with (ROOT / 'observations.csv').open() as f:
        times = {datetime.fromisoformat(r['time_lst']) for r in csv.DictReader(f)}
    history = np.array([all(t - h * HOUR in times for h in range(24)) for t in origins])
    legacy_train = train & six & history
    valid = complete & six & history & [cutoff <= t and t + H * HOUR < datetime(2025,10,1) for t in origins]
    base = x[:,:51]
    inputs = base[legacy_train]
    mean, scale = inputs.mean(axis=0), inputs.std(axis=0)
    z = (inputs - mean) / scale
    deltas = y[legacy_train,:96] - inputs[:, :1]
    bias = deltas.mean(axis=0)
    w = np.linalg.solve(z.T @ z + 100 * np.eye(z.shape[1]), z.T @ (deltas - bias))
    old = evaluate(base[valid], (mean, scale, w, bias)) + base[valid,:1]
    fits = dict(fit_candidates(x[train], y[train]))
    short = evaluate(x[valid], fits[selection['hourly_penalty']])[:, :96]
    lo = evaluate(x[valid], fits[selection['daily_penalty']])[:, 96:99]
    blend = np.clip((23 - np.arange(96)) / 12, 0, 1)
    new = old + blend * (short - old)
    old_days, new_days = [], []
    for i, t in enumerate(t for t, ok in zip(origins,valid) if ok):
        start = 23 - t.hour
        days = old[i,start:start+72].reshape(3,24)
        highs = days.max(axis=1)
        old_days.append(np.r_[days.min(axis=1),highs].round(1))
        lows = np.minimum(lo[i],highs)
        visible = new[i,start:12].round(1)
        if len(visible):
            lows[0] = min(lows[0],visible.min())
            highs[0] = max(highs[0],visible.max())
        new_days.append(np.r_[lows,highs].round(1))
    def report(h, d):
        err = abs(h - y[valid,:96])
        return dict(hourly_mae=[float(err[:,a:b].mean()) for a,b in BANDS],
                    daily_mae=abs(d - y[valid,96:]).mean(axis=0).tolist())
    result = dict(training_labels_before=str(cutoff), evaluation_labels_before='2025-10-01',
                  old_training_origins=int(legacy_train.sum()), new_training_origins=int(train.sum()),
                  scored_origins=int(valid.sum()), original_recipe=report(old,old_days),
                  upgraded_recipe=report(new,new_days))
    (ROOT / 'upgrade' / 'season_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
