"""Portable inference (Python standard library only). All times fixed UTC-5."""
import csv,gzip,json,math,struct
from datetime import datetime,timedelta
from pathlib import Path
from features import make_features
ROOT=Path(__file__).resolve().parent

def load_model(path=None):
 with gzip.open(path or ROOT/'model.json.gz','rt') as f:return json.load(f)

def predict_vector(x,model):
 if model['model_type'].startswith('ridge'):
  z=[(v-m)/s for v,m,s in zip(x,model['mean'],model['scale'])]
  delta=[b+sum(w*v for w,v in zip(weights,z)) for b,weights in zip(model['intercept'],model['weights'])]
 else:
  # sklearn trees compare float32 input values.
  xf=[struct.unpack('f',struct.pack('f',v))[0] for v in x]
  delta=[0.]*(model['horizon_hours']*3)
  for t in model['trees']:
   node=0
   while t['left'][node]!=-1:
    node=t['left'][node] if xf[t['feature'][node]]<=t['threshold'][node] else t['right'][node]
   for i,v in enumerate(t['values'][node]):delta[i]+=v/len(model['trees'])
  delta=[v*s+m for v,s,m in zip(delta,model['target_scale'],model['target_mean'])]
 return [[delta[h*3+k]+x[k] for k in range(3)] for h in range(model['horizon_hours'])]

def forecast(rows,model,reference_time=None):
 by={datetime.fromisoformat(r['time_lst']):r for r in rows}
 if not by:raise ValueError('No observations')
 now=max(by)
 if reference_time is not None:
  age=(reference_time-now).total_seconds()
  if age<0 or age>5400:raise ValueError('Latest hourly input is in the future or over 90 minutes old')
 x=make_features(by,now);pred=predict_vector(x,model)
 hourly=[]
 for h,(t,rh,p) in enumerate(pred,1):
  rh=max(0,min(100,rh));drop=p-x[2]
  sky=('Risque de précipitations' if rh>=85 and drop<=-0.3 else 'Nuageux' if rh>=75 else 'Partiellement nuageux' if rh>=55 else 'Possiblement dégagé')
  hourly.append(dict(time_lst=str(now+timedelta(hours=h)),temperature_c=round(t,1),humidity_pct=round(rh,1),pressure_kpa=round(p,2),sky_estimate=sky))
 daily=[]
 for d in range(1,4):
  date=(now+timedelta(days=d)).date();day=[r for r in hourly if datetime.fromisoformat(r['time_lst']).date()==date]
  if len(day)!=24:raise ValueError('Incomplete calendar day forecast')
  labels=[r['sky_estimate'] for r in day]
  daily.append(dict(date=str(date),min_temperature_c=min(r['temperature_c'] for r in day),max_temperature_c=max(r['temperature_c'] for r in day),sky_estimate=max(dict.fromkeys(labels),key=labels.count),hours=24))
 return dict(issued_at_lst=str(now),experimental=True,sky_method='humidity/pressure heuristic; not calibrated probability',next_12_hours=hourly[:12],next_3_days=daily)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('csv',nargs='?',default=str(ROOT/'observations.csv'));parser.add_argument('--live',action='store_true',help='Reject stale observations instead of historical replay');args=parser.parse_args()
 with open(args.csv,encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 from datetime import timezone
 reference=datetime.now(timezone(timedelta(hours=-5))).replace(tzinfo=None) if args.live else None
 print(json.dumps(forecast(rows,load_model(),reference),ensure_ascii=False,indent=2))
