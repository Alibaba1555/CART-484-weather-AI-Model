"""Frozen chronological protocol: train through Aug, validate Sep, test Oct-Nov."""
import csv,json,gzip,time
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from features import make_features,KEYS
ROOT=Path(__file__).resolve().parent
H=96
by={};rejected=0
for p in sorted((ROOT/'raw').glob('*.csv')):
 with p.open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):
   if r['Date/Time (LST)']>='2026-11-11':continue
   try:
    values=[float(r[k]) for k in ['Temp (°C)','Rel Hum (%)','Stn Press (kPa)']]
    if not all(np.isfinite(values)) or not(-60<=values[0]<=60 and 0<=values[1]<=100 and 80<=values[2]<=110):raise ValueError()
    stamp=datetime.fromisoformat(r['Date/Time (LST)'])
    by[stamp]=dict(zip(KEYS,values))
   except (ValueError,TypeError):rejected+=1
with (ROOT/'observations.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['time_lst']+KEYS)
 for t in sorted(by):w.writerow([str(t)]+[by[t][k] for k in KEYS])
origins=[];xs=[];ys=[];seasonal=[]
for now in sorted(by):
 # six-hour issuance cadence avoids excessively repetitive training rows.
 if now.hour%6:continue
 try:x=make_features(by,now)
 except ValueError:continue
 y=np.full((H,3),np.nan)
 for h in range(1,H+1):
  if now+timedelta(hours=h) in by:y[h-1]=[by[now+timedelta(hours=h)][k] for k in KEYS]
 if not np.isfinite(y[:72]).all():continue
 # repeat last 24 hours at matching clock hour; use actual preceding 24h only.
 try:s=[[by[now-timedelta(hours=(24-h%24)%24)][k] for k in KEYS] for h in range(1,H+1)]
 except KeyError:continue
 origins.append(now);xs.append(x);ys.append(y);seasonal.append(s)
X=np.array(xs);Y=np.array(ys);seasonal=np.array(seasonal)
valid=np.isfinite(Y).all(axis=(1,2))
train=np.array([t+timedelta(hours=H)<datetime(2026,9,1) for t in origins])&valid
val=np.array([t>=datetime(2026,9,1) and t+timedelta(hours=H)<datetime(2026,10,1) for t in origins])&valid
test=np.array([t>=datetime(2026,10,1) for t in origins])
# Target each of 96 future hours directly, as departure from current observation.
base=X[:,:3];delta=(Y-base[:,None,:]).reshape(len(Y),-1)
sc=StandardScaler().fit(X[train]);Xs=sc.transform(X)
candidates={}
for alpha in [10.,100.,1000.]:
 name=f'ridge_{int(alpha)}';m=Ridge(alpha=alpha).fit(Xs[train],delta[train]);p=m.predict(Xs[val]).reshape(-1,H,3)+base[val,None,:]
 score=float(np.mean(abs(p[:,:72,0]-Y[val,:72,0])))
 candidates[name]=(score,m);print('validation',name,score,flush=True)
# Bound forest size so portable inference remains suitable for the Pi.
forest=ExtraTreesRegressor(n_estimators=64,max_leaf_nodes=128,min_samples_leaf=12,max_features=0.8,random_state=484,n_jobs=2)
# Standardize target dimensions so humidity does not dominate tree splits.
ymean=delta[train].mean(axis=0);ystd=delta[train].std(axis=0);ystd[ystd==0]=1
forest.fit(X[train],(delta[train]-ymean)/ystd)
p=(forest.predict(X[val])*ystd+ymean).reshape(-1,H,3)+base[val,None,:]
candidates['extra_trees']=(float(np.mean(abs(p[:,:72,0]-Y[val,:72,0]))),forest)
print('validation extra_trees',candidates['extra_trees'][0],flush=True)
name,(score,model)=min(candidates.items(),key=lambda v:v[1][0])
if name=='extra_trees':pred=(model.predict(X[test])*ystd+ymean).reshape(-1,H,3)+base[test,None,:]
else:pred=model.predict(Xs[test]).reshape(-1,H,3)+base[test,None,:]
truth=Y[test];persist=np.broadcast_to(base[test,None,:],truth.shape);repeat=seasonal[test]
# Exact old model evaluated on identical inputs/origins, without retraining it.
old=json.loads((ROOT/'previous_model.json').read_text())
oldpred=[]
for i in np.where(test)[0]:
 now=origins[i];v=[]
 for lag in [0,1,3,24]:v.extend(by[now-timedelta(hours=lag)][k] for k in KEYS)
 v.extend([np.sin(2*np.pi*now.hour/24),np.cos(2*np.pi*now.hour/24)])
 z=(np.array(v)-old['mean'])/old['scale']
 oldpred.append((np.array(old['weights'])@z+old['intercept']).reshape(72,3))
oldpred=np.array(oldpred)
def errors(p,a):
 return dict(mae_c=float(np.nanmean(abs(p-a))),rmse_c=float(np.sqrt(np.nanmean((p-a)**2))))
def groups(p,a):
 return {n:errors(p[:,lo:hi,0],a[:,lo:hi,0]) for n,lo,hi in [('hours_1_12',0,12),('hours_1_24',0,24),('hours_25_48',24,48),('hours_49_72',48,72)]}
legacy=np.array([datetime(2026,11,4)<=origins[i]<=datetime(2026,11,7,23) for i in np.where(test)[0]])
report=dict(data_start=str(min(by)),data_end=str(max(by)),observations=len(by),rejected_rows=rejected,training_origins=int(train.sum()),validation_origins=int(val.sum()),test_origins=int(test.sum()),test_first=str(min(t for t,m in zip(origins,test) if m)),test_last=str(max(t for t,m in zip(origins,test) if m)),selected_model=name,validation_mae={k:v[0] for k,v in candidates.items()},test=dict(new_model=groups(pred,truth),persistence=groups(persist,truth),repeat_yesterday=groups(repeat,truth)),legacy_window=dict(origins=int(legacy.sum()),new_model=groups(pred[legacy],truth[legacy]),old_model=groups(oldpred[legacy],truth[legacy])))
# Old model saw part of Oct/Nov: ONLY compare on its original held-out Nov4-7 window.
(ROOT/'metrics.json').write_text(json.dumps(report,indent=2))
export=dict(schema_version=2,model_type=name,horizon_hours=H,timezone='UTC-05:00',last_training_target_before='2026-09-01',features_version=2)
if name.startswith('ridge'):
 export.update(mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),weights=model.coef_.tolist(),intercept=model.intercept_.tolist())
else:
 trees=[]
 for est in model.estimators_:
  t=est.tree_;trees.append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),values=t.value[:,:,0].tolist()))
 export.update(trees=trees,target_mean=ymean.tolist(),target_scale=ystd.tolist())
with gzip.open(ROOT/'model.json.gz','wt') as f:json.dump(export,f,separators=(',',':'))
np.savez_compressed(ROOT/'verification.npz',x=X[test],expected=pred)
with (ROOT/'test_predictions.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['origin_lst','horizon_h','actual_c','predicted_c','persistence_c','repeat_yesterday_c'])
 for row,i in enumerate(np.where(test)[0]):
  for h in range(72):w.writerow([str(origins[i]),h+1,truth[row,h,0],pred[row,h,0],persist[row,h,0],repeat[row,h,0]])
print(json.dumps(report,indent=2),flush=True)
