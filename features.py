import math
from datetime import datetime,timedelta
LAGS=[0,1,3,6,12,24,48,72]
KEYS=['temperature_c','humidity_pct','pressure_kpa']
def make_features(by,now):
    out=[]
    for lag in LAGS:
        row=by.get(now-timedelta(hours=lag))
        if row is None: raise ValueError('Missing hourly input at lag %s hours'%lag)
        v=[float(row[k]) for k in KEYS]
        if not all(math.isfinite(x) for x in v):raise ValueError('Non-finite input')
        out.extend(v)
    for lag in [1,3,6,12,24,48,72]:
        row=by[now-timedelta(hours=lag)]
        out.extend(float(by[now][k])-float(row[k]) for k in KEYS)
    phase=2*math.pi*((now-datetime(now.year,1,1)).total_seconds()/(86400*365.2425))
    hour=2*math.pi*now.hour/24
    out.extend([math.sin(hour),math.cos(hour),math.sin(phase),math.cos(phase),math.sin(2*phase),math.cos(2*phase)])
    return out
