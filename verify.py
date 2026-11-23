"""Verify exported inference and calendar-day aggregation; add daily metrics."""
import csv,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
from baseline_forecast import load_model,predict_vector,forecast
ROOT=Path(__file__).resolve().parent
model=load_model();v=np.load(ROOT/'verification.npz')
actual=np.array([predict_vector(x,model) for x in v['x']])
assert np.allclose(actual,v['expected'],atol=1e-8)
with (ROOT/'observations.csv').open() as f: rows=list(csv.DictReader(f))
by={datetime.fromisoformat(r['time_lst']):r for r in rows}
with (ROOT/'test_predictions.csv').open() as f:origins=sorted({datetime.fromisoformat(r['origin_lst']) for r in csv.DictReader(f)})
# Check every hour of a day, including midnight/noon, to ensure all 3 calendar days are complete.
for hour in range(24):
 now=datetime(2026,10,10,hour)
 window=[by[t] for t in sorted(by) if now-timedelta(hours=72)<=t<=now]
 result=forecast(window,model,now)
 assert len(result['next_12_hours'])==12 and all(d['hours']==24 for d in result['next_3_days'])
 assert result['next_3_days'][-1]['date']==str((now+timedelta(days=3)).date())
try:forecast([rows[-1]],model)
except ValueError:pass
else:raise AssertionError('Missing history should be rejected')
try:forecast(rows,model,max(by)+timedelta(hours=2))
except ValueError:pass
else:raise AssertionError('Stale input should be rejected')
# Daily extrema measured against actual calendar days, only where full observations exist.
daily={str(d):{'min_errors':[],'max_errors':[]} for d in [1,2,3]}
for i,now in enumerate(origins):
 for d in [1,2,3]:
  date=(now+timedelta(days=d)).date()
  hs=[h for h in range(1,97) if (now+timedelta(hours=h)).date()==date]
  stamps=[now+timedelta(hours=h) for h in hs]
  if len(hs)!=24 or any(t not in by for t in stamps):continue
  observed=[float(by[t]['temperature_c']) for t in stamps];pred=[actual[i,h-1,0] for h in hs]
  daily[str(d)]['min_errors'].append(abs(min(pred)-min(observed)))
  daily[str(d)]['max_errors'].append(abs(max(pred)-max(observed)))
report=json.loads((ROOT/'metrics.json').read_text())
report['daily_extrema']={d:dict(count=len(v['min_errors']),min_temperature_mae_c=float(np.mean(v['min_errors'])),max_temperature_mae_c=float(np.mean(v['max_errors']))) for d,v in daily.items()}
report['verification']={'all_exported_predictions_match_training_library':True,'every_issuance_hour_has_three_complete_calendar_days':True,'missing_history_and_stale_inputs_rejected':True}
(ROOT/'metrics.json').write_text(json.dumps(report,indent=2))
(ROOT/'baseline_example_forecast.json').write_text(json.dumps(forecast(rows,model),ensure_ascii=False,indent=2))
# Compact input example: exactly latest 73 hourly slots including endpoints.
with (ROOT/'example_input.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows([by[t] for t in sorted(by) if t>=max(by)-timedelta(hours=72)])
print(json.dumps(report['daily_extrema'],indent=2));print('Verification passed')
