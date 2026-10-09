"""Independent quote/label checks against the existing replay and residual-market diagnostic."""
import json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from analyze import DEST, CACHE, corr, avg, target, partitions, feature_matrix
sys.path.insert(0,str(DEST.parent))
from engine import Simulator, ms_to_stamp
r=json.loads((DEST/'results.json').read_text());checks=[]
for symbol in [601688,688981,601398]:
 raw=np.load(CACHE/f'{symbol}_v1.npz')['raw'];s=Simulator(symbol=symbol)
 for sec in [34800,46830,50700,53820]:
  s.advance(ms_to_stamp(sec*1000));row=raw[raw[:,0]==sec][0]
  bid=s.book.top(66,1)[0];ask=s.book.top(83,1)[0]
  assert tuple(row[2:6])==(bid[0],ask[0],bid[1],ask[1])
  checks.append(f'{symbol} {sec}: current quote exactly matches original engine')
 X,mid=feature_matrix(raw)
 for h in [1,30,300]:
  y,valid,safe=target(raw,mid,h);i=np.flatnonzero((raw[:,0]==50700)&valid)[0]
  s=Simulator(symbol=symbol);s.advance(ms_to_stamp((50700+h)*1000));future_mid=(s.book.top(66,1)[0][0]+s.book.top(83,1)[0][0])/2
  assert abs(y[i]-(future_mid/mid[i]-1)*10000)<1e-9
  checks.append(f'{symbol} {h}s forward label exactly matches original engine')
# Cross-stock shared future price movement is an outcome adjustment, never an input.
tab=pq.read_table(DEST/'factor_samples.parquet',columns=['symbol','second','mid_price','quote_age_seconds','obi1','trade5','future_1s_bp','future_30s_bp'])
arrays={k:tab[k].to_numpy() for k in tab.column_names};seconds=np.unique(arrays['second']);sid=np.searchsorted(seconds,arrays['second'])
robust=[]
for h in [1,30]:
 y=arrays[f'future_{h}s_bp'];finite=np.isfinite(y);sums=np.bincount(sid[finite],weights=y[finite],minlength=len(seconds));count=np.bincount(sid[finite],minlength=len(seconds))
 mean=np.divide(sums,count,out=np.zeros_like(sums),where=count>0);neutral=y-mean[sid]
 for key in ['obi1','trade5']:
  orig=[];adjusted=[]
  for symbol in r['metadata']['symbols']:
   mask=(arrays['symbol']==symbol)&(arrays['second']>=50430)&(arrays['second']+h<=53820)&finite&(arrays['quote_age_seconds']<=30)
   orig.append(corr(arrays[key][mask],y[mask]));adjusted.append(corr(arrays[key][mask],neutral[mask]))
  robust.append(dict(factor=key,horizon=h,original_raw_ic=avg(orig),market_adjusted_raw_ic=avg(adjusted)))
output=dict(independent_checks=checks,unit_checks=7,market_adjusted_diagnostic=robust,
 note='扣除同秒50只股票等权未来价变，只是事后目标诊断，不作为因子；仍是同一日尾段。')
(DEST/'verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
print(json.dumps(output,ensure_ascii=False))
