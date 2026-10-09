"""Render scientific figures and an offline interactive report from results.json."""
import csv,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pyarrow.parquet as pq

DEST=Path(__file__).resolve().parent
font_path = Path(os.environ.get('SSE_L2_FONT', '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
if font_path.is_file():
 font_manager.fontManager.addfont(str(font_path))
 font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
else:
 font_family = 'DejaVu Sans'
plt.rcParams.update({'font.family':font_family,'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f5f7fa','axes.facecolor':'#ffffff','savefig.facecolor':'#f5f7fa'})
r=json.loads((DEST/'results.json').read_text());m=r['metadata'];features=m['features'];names={k:n for k,n,_ in features};names['ridge']='12因子组合（岭回归）'
lookup={(v['factor'],v['horizon']):v for v in r['summary']};hs=m['horizons'];winner=m['selected_factor'];best=lookup[winner,30]
keys=[f[0] for f in features]
fig=plt.figure(figsize=(15,11),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1.3,1])
ax=fig.add_subplot(gs[0,0]);matrix=np.array([[lookup[k,h]['test']['raw_ic'] for h in hs] for k in keys]);bound=max(.15,float(np.max(np.abs(matrix))))
im=ax.imshow(matrix,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
ax.set_yticks(range(len(keys)),[names[k] for k in keys]);ax.set_xticks(range(len(hs)),[f'{h}秒' for h in hs]);ax.set_title('尾段留出：原始因子与未来中间价变化的相关系数',loc='left',pad=14)
for i in range(len(keys)):
 for j in range(len(hs)):
  v=matrix[i,j];ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(v)>bound*.60 else '#243b53')
fig.colorbar(im,ax=ax,shrink=.8,label='逐股 Pearson IC 等权均值')
ax=fig.add_subplot(gs[0,1]);x=np.arange(1,6)
for split,label,color in [('validation','验证段','#889db0'),('test','尾段留出','#1f6b89')]:
 vals=[b['mean'] for b in best[split]['bins']];ax.plot(x,vals,'o-',label=label,color=color,linewidth=2)
ax.axhline(0,color='#bccbd6',lw=.8);ax.set_xticks(x,[f'Q{i}' for i in x]);ax.set_ylabel('未来30秒中间价变化（bp）');ax.set_xlabel('因子从低到高；分组阈值只用上午训练段')
ax.set_title(f'验证段选出的候选：{names[winner]}',loc='left',pad=14);ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
ax=fig.add_subplot(gs[1,0])
for key,label,color in [('obi1','一档盘口量差','#1f6b89'),('ridge','12因子组合','#b27c46'),('mom5','过去5秒动量','#8798a7')]:
 ax.plot(hs,[lookup[key,h]['test']['ic'] for h in hs],'o-',label=label,color=color,lw=2)
ax.set_xscale('log');ax.set_xticks(hs,[str(h) for h in hs]);ax.set_xlabel('预测期限（秒，对数轴）');ax.set_ylabel('预测值与未来变化的 IC');ax.axhline(0,color='#bccbd6',lw=.8)
ax.set_title('预测关联随时间尺度如何变化',loc='left',pad=14);ax.legend(frameon=False,fontsize=9);ax.grid(axis='y',alpha=.15)
ax=fig.add_subplot(gs[1,1])
for key,label,color in [('volume5','近5秒成交量','#1f6b89'),('spread','当前价差','#b27c46')]:
 ax.plot(hs,[lookup[key,h]['test']['abs_ic'] for h in hs],'o-',label=label,color=color,lw=2)
ax.set_xscale('log');ax.set_xticks(hs,[str(h) for h in hs]);ax.set_xlabel('预测期限（秒，对数轴）');ax.set_ylabel('因子与未来绝对价格变化的 IC');ax.axhline(0,color='#bccbd6',lw=.8)
ax.set_title('成交量与价差：更适合判断波动大小吗？',loc='left',pad=14);ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
fig.suptitle('上交所 Level-2 日内因子探索 · 2026-09-23 · 50只股票',fontsize=17,x=.02,ha='left')
fig.supxlabel('上午训练 → 下午前段验证 → 14:00:30后留出测试。逐秒标签存在重叠；仅单日探索，相关性与分组价变均不等于可交易收益。',fontsize=10)
fig.savefig(DEST/'overview.png',dpi=160);fig.savefig(DEST/'overview.pdf');plt.close(fig)
# Smaller independent distribution plot for the preselected 30s candidate.
stockrows=[s for s in r['stocks'] if s['factor']==winner and s['horizon']==30]
vals=np.array([s['test']['ic'] for s in stockrows]);order=np.argsort(vals)
fig,ax=plt.subplots(figsize=(14,5),layout='constrained');ax.bar(np.arange(len(order)),vals[order],color=['#1f6b89' if v>=0 else '#b27c46' for v in vals[order]])
ax.set_xticks(np.arange(len(order)),[str(stockrows[i]['symbol']) for i in order],rotation=90,fontsize=8);ax.axhline(0,color='#8d9ca8',lw=.8)
ax.set_ylabel('尾段30秒预测IC');ax.set_title(f'{names[winner]}：正向预测关联股票 {sum(vals>0)} / {len(vals)}（验证段选因子，尾段展示差异）',loc='left');fig.savefig(DEST/'stock_consistency.png',dpi=160);plt.close(fig)
# Embed only the summary and essential per-stock metrics, avoiding duplicated large arrays.
compact=dict(metadata=m,summary=r['summary'],stocks=[] ,cases={},verification=json.loads((DEST/'verification.json').read_text()) if (DEST/'verification.json').exists() else None)
for s in r['stocks']:
 compact['stocks'].append(dict(symbol=s['symbol'],horizon=s['horizon'],factor=s['factor'],
   **{split:{k:s[split][k] for k in ['ic','raw_ic','accuracy','baseline_accuracy','moving_rate','r2','n']} for split in ['validation','test']},
   nonoverlap_ic=s['test']['nonoverlap_ic'],nonoverlap_n=s['test']['nonoverlap_n']))
for symbol in [601688,688981,601398]:
 tab=pq.read_table(DEST/'factor_samples.parquet',filters=[('symbol','=',symbol),('second','>=',50700),('second','<=',51000)],
        columns=['second','mid_price','future_30s_bp','obi1','trade5','ofi5'])
 a=np.column_stack([tab[c].to_numpy() for c in tab.column_names]);compact['cases'][str(symbol)]=np.round(a,6).tolist()
payload=json.dumps(compact,ensure_ascii=False,allow_nan=False,separators=(',',':'))
html=(DEST/'report_template.html').read_text().replace('__DATA__',payload)
(DEST/'report.html').write_text(html)
print(json.dumps({'winner':winner,'validation_ic':best['validation']['ic'],'test':best['test'],'report_bytes':len(html.encode()),'total_rows':m['grid_rows']},ensure_ascii=False))
