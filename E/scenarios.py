"""检查有预报、污染峰值以及观测稀疏时的误差变化。"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from E.evaluate import PRED,ROOT,OUT,read

def main():
    origins=read("origins")
    air=np.load(ROOT/"cache/A/air.npy",mmap_mode="r")
    weather=np.load(ROOT/"cache/B/forecast.npy",mmap_mode="r")
    cutoff=float(np.nanquantile(air[:731*24,:,0],0.90))
    pm=np.isfinite(air[:,:,0]).astype(np.int32)
    cumulative=np.vstack([np.zeros_like(pm[:1]),pm.cumsum(0,dtype=np.int32)])
    counts=cumulative[origins+1]-cumulative[origins-47]
    truth=read("truth")
    labels=["B1","B2",*[f"{m}_{s}" for m in ["M1","M2","A1","A2","A3"] for s in [17,29,43]]]
    arrays={m:read(m) for m in labels}
    focus=["B1",*[f"{m}_{s}" for m in ["A1","A2"] for s in [17,29,43]]]
    stats={}
    for i in range(0,len(origins),32):
        sl=slice(i,i+32)
        y=truth[sl]
        valid=np.isfinite(y)
        for pred in arrays.values():valid &= np.isfinite(pred[sl])
        has=np.isfinite(weather[origins[sl],:,:,0])
        for group,mask in [("可匹配预报",has),("无匹配预报",~has),("训练期P90及以上",y>=cutoff),("训练期P90以下",y<cutoff),("历史PM25有效率不足80%",counts[sl,:,None]<48*0.8),("历史PM25有效率至少80%",counts[sl,:,None]>=48*0.8)]:
            take=valid&mask
            for name in focus:
                row=stats.setdefault((group,name),[0,0.])
                row[0]+=int(take.sum())
                row[1]+=np.abs(arrays[name][sl]-y,where=take,out=np.zeros_like(y)).sum(dtype=np.float64)
    rows=[[g,m,n,err/n if n else np.nan,cutoff] for (g,m),(n,err) in stats.items()]
    pd.DataFrame(rows,columns=["group","model_seed","n","MAE","train_PM25_p90"]).to_csv(OUT/"scenarios.csv",index=False)
    df=pd.read_csv(OUT/"summary.csv")
    h=df[(df.city=="全区域")&(df.horizon>0)].pivot(index="horizon",columns="model",values="MAE")
    fig,ax=plt.subplots(figsize=(8,3.8),layout="constrained")
    ax.axhline(0,color="#484848",lw=.9)
    for u,v,c,mark,label in [("M2","A1","#0072B2","o","加入历史气象"),("A1","A2","#D55E00","^","再加入已发布预报"),("A3","A2","#008569","s","加入邻站连接")]:
        ax.plot(h.index,h[u]-h[v],color=c,marker=mark,markevery=3,label=label)
    ax.set(xlabel="预测时距（小时）",ylabel="MAE 降幅（正值为改善）",xticks=[1,3,6,12,18,24])
    ax.legend(frameon=False,ncol=3)
    fig.savefig(OUT/"ablation.png",dpi=180)
    plt.close(fig)
if __name__=="__main__":main()

