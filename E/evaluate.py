"""从预测数组重建各模型共同样本上的指标、案例和图。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"E/results"
PRED=ROOT/"cache/predictions"
plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"axes.unicode_minus":False,"axes.spines.right":False,"axes.spines.top":False})
NAMES={"B1":"持久性","B2":"日周期","M1":"TCN","M2":"STGCN","A1":"加入历史天气","A2":"加入历史天气和预报","A3":"只保留自环"}
COLORS=["#545b60","#a9a29a","#0072B2","#D55E00","#8268a4","#008569","#A85B7D"]
def read(name): return np.load(PRED/(name+".npy"),mmap_mode="r")

def main():
    stations=pd.read_csv(ROOT/"cache/A/stations.csv",dtype={"station_id":str,"city_id":str})
    cities=pd.read_csv(ROOT/"JZH_Air_2014_2018/metadata/cities.csv",dtype=str).set_index("city_id").name_Chinese
    names=cities.loc[stations.city_id].to_numpy()
    y=read("truth")
    origins=read("origins")
    config=json.loads((ROOT/"E/settings.json").read_text(encoding="utf-8-sig"))
    labels=["B1","B2"]+[f"{e}_{s}" for e in config["experiments"] for s in config["seeds"]]
    predictions={n:read(n) for n in labels}
    shape=(len(stations),24)
    count=np.zeros(shape,dtype="int64")
    true_count=0
    summed={n:np.zeros((2,*shape)) for n in labels}
    cases={}
    for start in range(0,len(origins),64):
        sl=slice(start,start+64)
        truth=y[sl]
        valid=np.isfinite(truth)
        true_count+=int(valid.sum())
        for pred in predictions.values(): valid &= np.isfinite(pred[sl])
        count+=valid.sum(0)
        for n,pred in predictions.items():
            error=np.where(valid,pred[sl]-truth,0).astype("float64")
            summed[n][0]+=np.abs(error).sum(0)
            summed[n][1]+=np.square(error).sum(0)
        full=valid.all(2)
        if full.any():
            diff=np.abs(predictions["B1"][sl]-truth).mean(2)-np.abs(predictions["A2_17"][sl]-truth).mean(2)
            for title,fn,fill,sign in [("改善案例",np.argmax,-np.inf,1),("失败案例",np.argmin,np.inf,-1)]:
                scores=np.where(full,diff,fill)
                i,j=np.unravel_index(fn(scores),scores.shape)
                value=float(scores[i,j])
                if title not in cases or value*sign>cases[title][0]*sign:
                    cases[title]=(value,start+i,j)
    rows=[]
    for label in labels:
        model,_,seed=label.partition("_")
        for city in ["全区域",*sorted(set(names))]:
            sel=np.ones(len(names),bool) if city=="全区域" else names==city
            n=count[sel].sum(0)
            a,b=summed[label][:,sel].sum(1)
            for h in [0,*range(1,25)]:
                total,ae,se=(n.sum(),a.sum(),b.sum()) if h==0 else (n[h-1],a[h-1],b[h-1])
                if total:
                    rows.append([model,seed or "固定",city,h,int(total),ae/total,np.sqrt(se/total)])
    df=pd.DataFrame(rows,columns=["model","seed","city","horizon","n","MAE","RMSE"])
    df.to_csv(OUT/"metrics.csv",index=False)
    summary=df.groupby(["model","city","horizon"],as_index=False).agg(n=("n","first"),MAE=("MAE","mean"),MAE_sd=("MAE","std"),RMSE=("RMSE","mean"),RMSE_sd=("RMSE","std"))
    summary.to_csv(OUT/"summary.csv",index=False)
    overall=summary[(summary.city=="全区域")&(summary.horizon>0)]
    fig,ax=plt.subplots(figsize=(8.6,4.5),layout="constrained")
    for idx,model in enumerate(["B1","B2","M1","M2","A2"]):
        g=overall[overall.model==model]
        ax.plot(g.horizon,g.MAE,label=NAMES[model],color=COLORS[[0,1,2,3,5][idx]],marker=["o","x","s","^","D"][idx],markevery=4,ls=["-","--",":","-.","-"][idx])
    ax.set(xlabel="预测时距（小时）",ylabel="MAE（原始浓度数值，越低越好）",xticks=[1,3,6,12,18,24])
    ax.legend(ncol=3,frameon=False,fontsize=9)
    fig.savefig(OUT/"horizons.png",dpi=180);plt.close(fig)
    report=[]
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout="constrained")
    for ax,(name,(delta,i,j)) in zip(axes,cases.items()):
        t=pd.Timestamp("2015-01-01")+pd.Timedelta(hours=int(origins[i]))
        for label,array,color,style in [("实测",y,"#273743","-"),("持久性",predictions["B1"],"#888888","--"),("气象融合",predictions["A2_17"],"#008569","-.")]:
            ax.plot(range(1,25),array[i,j],label=label,color=color,ls=style)
            for h,val in enumerate(array[i,j],1):
                report.append([name,stations.station_id.iloc[j],names[j],str(t),h,label,float(val),delta])
        ax.set(title=f"{name} | {names[j]} {stations.station_id.iloc[j]}\n{t:%Y-%m-%d %H:%M}",xlabel="起报后的小时",ylabel="PM2.5（原始浓度数值）")
    axes[0].legend(frameon=False);fig.savefig(OUT/"cases.png",dpi=180);plt.close(fig)
    pd.DataFrame(report,columns=["case","station_id","city","forecast_origin","horizon","series","value","mae_gain"]).to_csv(OUT/"cases.csv",index=False)
    subset=summary[(summary.city!="全区域")&(summary.horizon==0)].pivot(index="city",columns="model",values="MAE")
    gain=(subset["A1"]-subset["A2"]).sort_values()
    fig,ax=plt.subplots(figsize=(8.5,4.3),layout="constrained")
    ax.bar(range(len(gain)),gain,color=["#007f70" if g>=0 else "#bc623e" for g in gain])
    ax.set(xticks=range(len(gain)),xticklabels=gain.index,ylabel="加入预报后的 MAE 降幅",xlabel="城市（正值表示预报改善误差）");ax.tick_params(axis="x",labelrotation=55);ax.axhline(0,color="#404040",lw=.8)
    fig.savefig(OUT/"cities.png",dpi=180);plt.close(fig)
    (OUT/"coverage.json").write_text(json.dumps(dict(origins=len(origins),nodes=len(stations),true_labels=true_count,common_labels=int(count.sum()),all_cells=int(y.size)),indent=2))
    print(summary[(summary.city=="全区域")&(summary.horizon==0)].to_string(index=False))

if __name__=="__main__": main()

