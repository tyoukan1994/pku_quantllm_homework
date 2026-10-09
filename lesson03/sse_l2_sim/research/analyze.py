"""Point-in-time, single-day L2 factor study. Run from the simulator directory with .venv/bin/python research/analyze.py.
Only actual events <= observation time form features. Quotes never use snapshots.
"""
import argparse, csv, heapq, json, sys, time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import HistoricalBook, load_stock, cents, stamp_to_ms, ms_to_stamp, DEFAULT_DATA, DATE
DEST = Path(__file__).resolve().parent
CACHE = ROOT / '.cache' / 'factor_cache'
SESSIONS = [(34200,41400),(46800,53820)]
HORIZONS = [1,5,10,30,60,180,300]
FACTORS = [
 ('obi1','买卖一档量差','(买一量−卖一量)／(买一量＋卖一量)'),
 ('obi5','五档盘口量差','买卖各前五档数量之差／数量之和'),
 ('obi10w','十档加权量差','第 k 档权重 1/k，加权买卖数量之差／之和'),
 ('microdev','微价偏离','(卖一价×买一量＋买一价×卖一量)／总量，相对中间价的偏离（bp）；与一档量差有机械联系'),
 ('trade5','近5秒主动买卖压力','(主动买成交量−主动卖成交量)／已知方向成交量，窗口 (t−5,t]'),
 ('trade30','近30秒主动买卖压力','同上，窗口 (t−30,t]；无成交设为0'),
 ('ofi5','近5秒订单流不平衡','逐事件按最优买卖价及数量变化计算 OFI，近5秒之和／近5秒平均单边一档深度'),
 ('cancel5','近5秒撤单压力','(卖方撤单量−买方撤单量)／撤单总量，覆盖全订单簿'),
 ('new5','近5秒新增挂单压力','(买方新增剩余挂单量−卖方新增剩余挂单量)／总量；并非完整原始报单量'),
 ('mom5','近5秒价格动量','过去5秒中间价变化（bp），作为价格信息基线'),
 ('volume5','近5秒成交量','log(1＋过去5秒成交股数)，更适合判断波动强弱'),
 ('spread','买卖价差','(卖一价−买一价)／中间价×10000（bp）')]
NAMES = {k:n for k,n,_ in FACTORS}
RAW = ['sec','session','bid','ask','bq','aq','b5','a5','bw','aw','ofi','buy','sell','volume','cb','cs','nb','ns','age']


class FastBook(HistoricalBook):
    def __init__(self):
        super().__init__(); self.heaps={66:[],83:[]}
    def best(self,side):
        h=self.heaps[side]; levels=self.levels[side]
        while h and (-h[0] if side==66 else h[0]) not in levels: heapq.heappop(h)
        if not h:return (0,0)
        p=-h[0] if side==66 else h[0]
        return p,levels[p]
    def quote(self):
        return (*self.best(66),*self.best(83))
    def apply(self,typ,x):
        new = typ=='O' and x['OrderKind']==65
        if new:
            side=x['FunctionCode'];p=cents(x['Price']); absent=p not in self.levels[side]
        super().apply(typ,x)
        if new and absent:heapq.heappush(self.heaps[side],-p if side==66 else p)


def event_ofi(old,new):
    b0,qb0,a0,qa0=old; b1,qb1,a1,qa1=new
    if not min(b0,a0,b1,a1)>0:return 0
    return (qb1 if b1>=b0 else 0)-(qb0 if b1<=b0 else 0)-(qa1 if a1<=a0 else 0)+(qa0 if a1>=a0 else 0)


def extract(symbol):
    cached=CACHE/f'{symbol}_v1.npz'
    if cached.exists():return np.load(cached)['raw']
    events,_,preclose,excluded=load_stock(symbol)
    b=FastBook();i=0;rows=[];last_change=None
    for session,(start,end) in enumerate(SESSIONS):
        for sec in range(start,end+1):
            target=ms_to_stamp(sec*1000);agg=np.zeros(9)
            while i<len(events) and events[i][2]['Time']<=target:
                _,typ,x=events[i];old=b.quote();b.apply(typ,x);new=b.quote();i+=1
                if old!=new or typ=='T':last_change=stamp_to_ms(x['Time'])/1000
                event_sec=stamp_to_ms(x['Time'])/1000
                # Reset flow windows at each continuous session; exclude auctions.
                if event_sec<start:continue
                agg[0]+=event_ofi(old,new)
                if typ=='T':
                    v=x['Volume'];agg[3]+=v
                    if x['BSFlag']==66:agg[1]+=v
                    elif x['BSFlag']==83:agg[2]+=v
                elif x['OrderKind'] in (65,68):
                    index=(6 if x['OrderKind']==65 else 4)+(x['FunctionCode']==83)
                    agg[index+1]+=x['Volume']
            bid,bq,ask,aq=b.quote();bids=b.top(66);asks=b.top(83)
            b5=sum(q for _,q in bids[:5]);a5=sum(q for _,q in asks[:5])
            bw=sum(q/(k+1) for k,(_,q) in enumerate(bids));aw=sum(q/(k+1) for k,(_,q) in enumerate(asks))
            # agg[0:4]=ofi,buy,sell,volume; agg[5:9]=cancelB,cancelS,newB,newS
            age=sec-last_change if last_change is not None else 1e9
            rows.append([sec,session,bid,ask,bq,aq,b5,a5,bw,aw,*agg[:4],*agg[5:9],age])
    raw=np.asarray(rows,dtype=np.float64)
    CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(cached,raw=raw)
    anomaly=sum(b.stats[k] for k in ['missing_cancel','missing_trade','trade_overdraw','cancel_overdraw'])
    if anomaly:raise ValueError(f'{symbol}: reconstructed book anomalies {anomaly}')
    return raw


def rolling(x,window):
    c=np.concatenate(([0.],np.cumsum(x)));idx=np.arange(len(x));return c[idx+1]-c[np.maximum(0,idx+1-window)]


def ratio(a,b):return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)


def feature_matrix(raw):
    X=np.full((len(raw),len(FACTORS)),np.nan);mid=(raw[:,2]+raw[:,3])/2
    for session in [0,1]:
        ids=np.flatnonzero(raw[:,1]==session);r=raw[ids];m=mid[ids]
        bq,aq=r[:,4],r[:,5];spread=ratio(r[:,3]-r[:,2],m)*1e4
        b5,a5=r[:,6],r[:,7];bw,aw=r[:,8],r[:,9]
        t5b,t5s=rolling(r[:,11],5),rolling(r[:,12],5)
        t30b,t30s=rolling(r[:,11],30),rolling(r[:,12],30)
        cb,cs=rolling(r[:,14],5),rolling(r[:,15],5)
        nb,ns=rolling(r[:,16],5),rolling(r[:,17],5)
        depth=rolling((bq+aq)/2,5)/np.minimum(np.arange(len(r))+1,5)
        momentum=np.full(len(r),np.nan);momentum[5:]=ratio(m[5:]-m[:-5],m[:-5])*1e4
        X[ids]=np.column_stack([ratio(bq-aq,bq+aq),ratio(b5-a5,b5+a5),ratio(bw-aw,bw+aw),
             spread*ratio(bq-aq,bq+aq)/2,ratio(t5b-t5s,t5b+t5s),ratio(t30b-t30s,t30b+t30s),
             ratio(rolling(r[:,10],5),depth),ratio(cs-cb,cs+cb),ratio(nb-ns,nb+ns),
             momentum,np.log1p(rolling(r[:,13],5)),spread])
    return X,mid


def target(raw,mid,h):
    idx=np.arange(len(raw));future=idx+h;ok=future<len(raw);safe=np.minimum(future,len(raw)-1)
    ok &= (raw[safe,1]==raw[:,1])&(raw[safe,0]-raw[:,0]==h)
    ok &= (raw[:,2]>0)&(raw[:,3]>raw[:,2])&(raw[safe,2]>0)&(raw[safe,3]>raw[safe,2])
    y=np.full(len(raw),np.nan);y[ok]=(mid[safe[ok]]/mid[ok]-1)*1e4
    return y,ok,safe


def partitions(raw,h,valid):
    sec=raw[:,0];warm=((raw[:,1]==0)&(sec>=34230))|((raw[:,1]==1)&(sec>=46830))
    valid=valid&warm&(raw[:,18]<=30)
    return [valid&(sec>=lo)&(sec+h<=hi) for lo,hi in [(34230,41400),(46830,50400),(50430,53820)]]


def corr(x,y):
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])


def ranks(x):
    unique,inv,counts=np.unique(x,return_inverse=True,return_counts=True)
    end=np.cumsum(counts);return ((end-counts+end-1)/2)[inv]


def avg(vals):
    vals=[v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def transform_fit(X):
    lo,hi=np.quantile(X,[.01,.99],axis=0);clipped=np.clip(X,lo,hi)
    mu=clipped.mean(0);sd=clipped.std(0);sd[sd<1e-8]=1
    return lo,hi,mu,sd


def transform(X,params):
    lo,hi,mu,sd=params;return (np.clip(X,lo,hi)-mu)/sd


def fit_predict(xtrain,ytrain,xval,xtest,alpha=1):
    ym=ytrain.mean();b=np.linalg.solve(xtrain.T@xtrain/len(ytrain)+alpha*np.eye(xtrain.shape[1]),xtrain.T@(ytrain-ym)/len(ytrain))
    return xtrain@b+ym,xval@b+ym,xtest@b+ym,b


def evaluate(y,p,reference,rawfactor,edges,spread):
    changing=np.abs(y)>1e-9;n=len(y);nm=int(changing.sum())
    correct=int(((p[changing]>0)==(y[changing]>0)).sum())
    baseline=int(((reference>0)==(y[changing]>0)).sum())
    variance=float(np.mean((y-reference)**2))
    groups=np.searchsorted(edges,rawfactor,side='right')
    bins=[]
    for g in range(5):
        q=groups==g;ny=int(q.sum());mv=q&changing
        bins.append(dict(n=ny,mean=avg(y[q]),up_rate=float(np.mean(y[mv]>0)) if mv.any() else None,
                         mean_abs=avg(np.abs(y[q])),mean_half_spread=avg(spread[q]/2)))
    return dict(n=n,moving_n=nm,moving_rate=nm/n,ic=corr(p,y),raw_ic=corr(rawfactor,y),rank_ic=corr(ranks(rawfactor),ranks(y)),
        correct=correct,baseline_correct=baseline,accuracy=correct/nm if nm else None,
        baseline_accuracy=baseline/nm if nm else None,r2=1-float(np.mean((p-y)**2))/variance if variance>0 else None,
        abs_ic=corr(rawfactor,np.abs(y)),bins=bins)


def merge_metrics(rows):
    out={k:avg([r[k] for r in rows]) for k in ['ic','raw_ic','rank_ic','r2','abs_ic']}
    n=sum(r['n'] for r in rows);moving=sum(r['moving_n'] for r in rows)
    out.update(n=n,moving_n=moving,moving_rate=moving/n if n else 0,
        accuracy=sum(r['correct'] for r in rows)/moving if moving else None,
        baseline_accuracy=sum(r['baseline_correct'] for r in rows)/moving if moving else None,
        positive_stocks=sum(r['ic'] is not None and r['ic']>0 for r in rows),stocks=len(rows))
    out['bins']=[dict(n=sum(r['bins'][g]['n'] for r in rows),
                **{k:avg([r['bins'][g][k] for r in rows]) for k in ['mean','up_rate','mean_abs','mean_half_spread']}) for g in range(5)]
    return out


def run(symbols):
    started=time.time();stock_results=[];feature_counts=[];raw_archive=[];timeseries={};all_rows=[]
    for number,symbol in enumerate(symbols):
        raw=extract(symbol);X,mid=feature_matrix(raw);validX=np.isfinite(X).all(1)
        feature_counts.append(dict(symbol=int(symbol),grid_rows=len(raw),quote_valid=int(((raw[:,2]>0)&(raw[:,3]>raw[:,2])).sum()),stale_rows=int((raw[:,18]>30).sum())))
        if symbol in (601688,688981,601398):
            take=(raw[:,0]>=34800)&(raw[:,0]<=35100) # 09:40-09:45
            timeseries[str(symbol)]={'columns':['second','mid','obi1','trade5','ofi5','spread'],
                'rows':np.column_stack([raw[take,0],mid[take]/100,X[take,0],X[take,4],X[take,6],X[take,11]]).round(6).tolist()}
        # Save causal features and aligned future labels in one auditable parquet.
        cols={'symbol':np.repeat(symbol,len(raw)).astype('int32'),'second':raw[:,0].astype('int32'),'session':raw[:,1].astype('int8'),
              'mid_price':mid/100,'bid_price':raw[:,2]/100,'ask_price':raw[:,3]/100,'quote_age_seconds':raw[:,18]}
        cols.update({k:X[:,j] for j,(k,_,_) in enumerate(FACTORS)})
        for h in HORIZONS:
            y,valid,_=target(raw,mid,h);cols[f'future_{h}s_bp']=y
            masks=partitions(raw,h,valid&validX);train,val,test=masks
            if min(map(np.sum,masks))<100:raise ValueError(f'{symbol}: insufficient samples')
            params=transform_fit(X[train]);xt,xv,xe=[transform(X[m],params) for m in masks]
            yt,yv,ye=[y[m] for m in masks];reference=yt.mean()
            for j,key in enumerate([k for k,_,_ in FACTORS]+['ridge']):
                sel=[j] if j<len(FACTORS) else list(range(len(FACTORS)))
                ptr,pv,pe,b=fit_predict(xt[:,sel],yt,xv[:,sel],xe[:,sel])
                source_train=X[train,j] if j<len(FACTORS) else ptr
                edges=np.quantile(source_train,[.2,.4,.6,.8])
                vsource=X[val,j] if j<len(FACTORS) else pv
                esource=X[test,j] if j<len(FACTORS) else pe
                # A majority-direction baseline is trained on moving morning labels.
                train_changes=yt[np.abs(yt)>1e-9];majority=1 if np.mean(train_changes>0)>=.5 else -1
                vm=evaluate(yv,pv,reference,vsource,edges,X[val,11]);em=evaluate(ye,pe,reference,esource,edges,X[test,11])
                vm['baseline_correct']=int(((yv[np.abs(yv)>1e-9]>0)==(majority>0)).sum());vm['baseline_accuracy']=vm['baseline_correct']/vm['moving_n'] if vm['moving_n'] else None
                em['baseline_correct']=int(((ye[np.abs(ye)>1e-9]>0)==(majority>0)).sum());em['baseline_accuracy']=em['baseline_correct']/em['moving_n'] if em['moving_n'] else None
                # Non-overlapping starts aligned to each horizon as a serial-dependence diagnostic.
                thin=(raw[test,0].astype(int)%h)==0
                em['nonoverlap_ic']=corr(pe[thin],ye[thin]);em['nonoverlap_n']=int(thin.sum())
                stock_results.append(dict(symbol=int(symbol),horizon=h,factor=key,train_n=int(train.sum()),
                     coefficient=b.round(8).tolist(),validation=vm,test=em))
            print(f'{symbol} {h}s: train={train.sum()} validation={val.sum()} test={test.sum()}',flush=True) if number==0 else None
        all_rows.append(pa.table(cols))
        print(f'[{number+1}/{len(symbols)}] {symbol}: {len(raw)} one-second observations, elapsed {time.time()-started:.1f}s',flush=True)
    pq.write_table(pa.concat_tables(all_rows),DEST/'factor_samples.parquet',compression='zstd')
    summaries=[]
    for h in HORIZONS:
        for key in [k for k,_,_ in FACTORS]+['ridge']:
            rs=[r for r in stock_results if r['horizon']==h and r['factor']==key]
            summaries.append(dict(factor=key,horizon=h,validation=merge_metrics([r['validation'] for r in rs]),test=merge_metrics([r['test'] for r in rs])))
    # Predeclared main horizon; candidate selection uses validation only.
    primary=[r for r in summaries if r['horizon']==30]
    winner=max(primary,key=lambda r:r['validation']['ic'] if r['validation']['ic'] is not None else -1)['factor']
    metadata=dict(date='2026-09-23',symbols=list(map(int,symbols)),horizons=HORIZONS,primary_horizon=30,
        selected_factor=winner,selection='按13:00:30—14:00验证段的30秒股票平均预测IC选一个模型；不使用尾段结果选模型',
        grid_rows=sum(r['grid_rows'] for r in feature_counts),grid_seconds=1,features=FACTORS,
        windows={'train':'09:30:30—11:30','validation':'13:00:30—14:00','test':'14:00:30—14:57'},
        label='未来h秒中间价相对当前中间价的变化，单位bp；1bp=0.01%',
        quality=feature_counts,elapsed_seconds=round(time.time()-started,1),
        caveats=['单日50只股票；尾段留出不等于跨日泛化验证','标签重叠、股票共振，不能把全部秒样本当成独立观测',
         '窗口末端按h秒清除跨界标签；因子窗口在午休重新开始；只用当前及过去逐笔事件',
         '输入1%/99%缩尾阈值、均值和标准差均只由上午训练段估计',
         '方向准确率只在未来价格发生变化的样本上计算，另报变化比例；基线为上午多数涨跌方向',
         '相关性均为逐股相关后等权平均；分组收益为逐股分组均值等权平均；准确率按变化样本合计',
         '未用未来快照对齐；字段由重建推断，业务时间并非实际行情接收时间',
         '微价偏离与一档量差机械相关；多因子并不代表多份独立信息',
         '仅做价格预测，未做可执行策略回测；价差、费用、延迟、市场冲击及A股T+1都会影响可交易性'])
    result=dict(metadata=metadata,summary=summaries,stocks=stock_results,timeseries=timeseries)
    (DEST/'results.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':')))
    with (DEST/'factor_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['factor','name','horizon','split','n','ic','raw_ic','rank_ic','r2','accuracy','baseline_accuracy','moving_rate','positive_stocks','stocks','abs_ic']
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in summaries:
            for split in ['validation','test']:
                w.writerow(dict(factor=r['factor'],name=NAMES.get(r['factor'],'12因子岭回归'),horizon=r['horizon'],split=split,
                    **{k:r[split][k] for k in fields[4:]}))
    print(json.dumps({'selected':winner,'seconds':metadata['elapsed_seconds'],'rows':metadata['grid_rows'],
        'main_test':[r for r in summaries if r['horizon']==30]},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--symbols',nargs='*',type=int);args=parser.parse_args()
    symbols=args.symbols or sorted(pq.read_table(DEFAULT_DATA/f'ord_{DATE}.parquet',columns=['Symbol'])['Symbol'].unique().to_pylist())
    run(symbols)
