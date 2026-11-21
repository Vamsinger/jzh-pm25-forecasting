"""发布一个可手工检查的上海站点 24 步预测长表。"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def main():
    cache=ROOT/"cache"
    pred=cache/"predictions"
    st=pd.read_csv(cache/"A/stations.csv",dtype=str)
    origins=np.load(pred/"origins.npy")
    load=lambda key:np.load(pred/(key+".npy"),mmap_mode="r")
    raw=np.load(cache/"A/air.npy",mmap_mode="r")
    weather=np.load(cache/"B/forecast.npy",mmap_mode="r")
    obs=np.load(cache/"B/met.npy",mmap_mode="r")
    truth=load("truth")
    daily=load("B2")
    sh=np.flatnonzero(st.city_id=="002")
    # 固定选取最早具备整段真值、日周期基线和至少一条预报的上海样本。
    for i,t in enumerate(origins):
        eligible=np.isfinite(truth[i,sh]).all(1)&np.isfinite(daily[i,sh]).all(1)&np.isfinite(weather[t,sh,:,0]).any(1)
        if eligible.any():
            j=sh[np.flatnonzero(eligible)[0]]
            break
    else: raise ValueError("没有符合条件的上海样本")
    base=pd.Timestamp("2015-01-01")
    time=base+pd.Timedelta(hours=int(t))
    out=ROOT/"E/results"
    history=pd.DataFrame(raw[t-47:t+1,j],columns=["PM25","PM10","NO2","CO","O3","SO2"])
    history.insert(0,"time",pd.date_range(end=time,periods=48,freq="h"))
    history.insert(0,"station_id",str(st.station_id.iloc[j]))
    history[["Temperature","Humidity","WindSpeed"]]=obs[t-47:t+1,j]
    history.to_csv(out/"shanghai_history.csv",index=False)
    rows=[]
    for model in ["B1","B2","M1_17","M2_17","A1_17","A2_17","A3_17"]:
        values=load(model)[i,j]
        for h,value in enumerate(values,1):
            fc=weather[t,j,h-1]
            published=time-pd.Timedelta(hours=float(fc[3])) if np.isfinite(fc[3]) else pd.NaT
            target=time+pd.Timedelta(hours=h)
            if pd.notna(published): assert published<=time
            rows.append([model,str(st.station_id.iloc[j]),time,h,target,float(value),float(truth[i,j,h-1]),published,*fc])
    df=pd.DataFrame(rows,columns=["model","station_id","forecast_origin","horizon","target_time","y_pred","y_true","forecast_issue_time","Up_Temperature","Bottom_Temperature","WindStrength","forecast_age_hours"])
    df.to_csv(out/"shanghai_predictions.csv",index=False)
    print(time,st.station_id.iloc[j],len(df),"行")
if __name__=="__main__":main()

