"""用验证年份的一个上海样本串起空气、天气、邻站与简单基线。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from C.models import baselines
from D.models import make_graph

def main():
    root=Path(__file__).resolve().parents[1]
    cache=root/"cache"
    st=pd.read_csv(cache/"A/stations.csv",dtype={"station_id":str,"city_id":str,"district_id":str})
    a=np.load(cache/"A/air.npy",mmap_mode="r")
    f=np.load(cache/"B/forecast.npy",mmap_mode="r")
    j=st.index[st.city_id=="002"][0]
    start=pd.Timestamp("2015-01-01")
    for t in range(731*24+48,1096*24-24):
        y=a[t+1:t+25,j,0]
        hist=a[t-47:t+1,j:j+1,0]
        if np.isfinite(y).all() and np.isfinite(hist).all() and np.isfinite(f[t,j,:,0]).any(): break
    else: raise ValueError("上海验证样本缺失")
    p,_=baselines(hist[None])
    err=np.abs(p[0,0]-y)
    np.testing.assert_allclose(err.mean(),sum(abs(float(v)-float(hist[-1,0])) for v in y)/24,rtol=1e-6)
    selected=json.loads((root/"E/results/frozen_M2.json").read_text())[0]["k"]
    graph=make_graph(st,selected)
    nb=np.flatnonzero(graph[j]>0)
    t0=start+pd.Timedelta(hours=t)
    out=pd.DataFrame({"station_id":st.loc[j,"station_id"],"forecast_origin":str(t0),"horizon":np.arange(1,25),"target_time":pd.date_range(t0+pd.Timedelta(hours=1),periods=24,freq="h"),"truth":y,"persistence":p[0,0],"absolute_error":err,"forecast_issue_time":[t0-pd.Timedelta(hours=int(age)) if np.isfinite(age) else pd.NaT for age in f[t,j,:,3]],"forecast_T_high":f[t,j,:,0]})
    assert (out.forecast_issue_time.dropna()<=t0).all()
    out.to_csv(root/"E/results/pilot_shanghai.csv",index=False)
    print("站点",st.loc[j,"station_id"],"起报",t0,"邻站",st.station_id.iloc[nb[nb!=j]].tolist(),"持久性 MAE",float(err.mean()))

if __name__=="__main__":main()

