"""Standard-library smoke tests for integration edge cases."""
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forecast import forecast, load_model, predict_vector
from features import make_features

with (ROOT / 'observations.csv').open() as f:
    rows = {datetime.fromisoformat(r['time_lst']):r for r in csv.DictReader(f)}
model = load_model()
for hour in range(24):
    now = datetime(2026,10,10,hour)
    sample = [rows[now - timedelta(hours=i)] for i in range(73)]
    result = forecast(reversed(sample), model, now)
    assert result == forecast(sample, model, now)
    assert len(result['next_12_hours']) == 12
    curve = predict_vector(make_features(rows,now), model)
    assert len(curve) == 96
    for i, r in enumerate(result['next_12_hours']):
        assert r['temperature_c'] == round(curve[i][0],1)
    for i, d in enumerate(result['next_3_days'],1):
        assert d['date'] == str((now + timedelta(days=i)).date()) and d['hours'] == 24
        assert d['min_temperature_c'] <= d['max_temperature_c']
        for r in result['next_12_hours']:
            if r['time_lst'][:10] == d['date']:
                assert d['min_temperature_c'] <= r['temperature_c'] <= d['max_temperature_c']
for invalid_rows, ref in [(sample[:2],now),(sample,now+timedelta(hours=2)),
                    (sample,now-timedelta(seconds=1)),
                    ([dict(sample[0],temperature_c='nan')] + sample[1:],now)]:
    try:
        forecast(invalid_rows, model, ref)
    except ValueError:
        continue
    raise AssertionError('Invalid forecast input accepted')
with (ROOT / 'observations.csv').open() as f:
    replay = forecast(csv.DictReader(f), model)
with (ROOT / 'pi' / 'example_input.csv').open() as f:
    assert replay == forecast(csv.DictReader(f),model)
assert replay == json.loads((ROOT / 'pi' / 'example_forecast.json').read_text())
print('All 24 issue hours, legacy vector API, input errors, daily bounds and streamed replay passed.')
