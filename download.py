from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import urllib.request,time,csv,io,json,hashlib
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw';RAW.mkdir(exist_ok=True)
def download(ym):
 y,m=ym;p=RAW/f'{y}-{m:02}.csv'
 url=f'https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=51157&Year={y}&Month={m}&Day=1&timeframe=1&submit=Download+Data'
 for attempt in range(3):
  try:
   b=p.read_bytes() if p.exists() else urllib.request.urlopen(url,timeout=50).read()
   r=list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
   assert r and 'Temp (°C)' in r[0] and r[0]['Station Name']=='MONTREAL INTL A',str(r[:1])[:100]
   p.write_bytes(b)
   return dict(file=p.name,url=url,sha256=hashlib.sha256(b).hexdigest(),rows=len(r))
  except Exception as e:
   if attempt==2:return dict(file=p.name,error=str(e))
   time.sleep(1)
jobs=[(y,m) for y in range(2023,2027) for m in range(1,13) if (y,m)<=(2026,11)]
results=[]
with ThreadPoolExecutor(max_workers=6) as pool:
 for f in as_completed([pool.submit(download,j) for j in jobs]):
  r=f.result();results.append(r);print(len(results),r['file'],r.get('error','OK'),flush=True)
(ROOT/'sources.json').write_text(json.dumps(sorted(results,key=lambda r:r['file']),indent=2))
assert not any('error' in r for r in results)
