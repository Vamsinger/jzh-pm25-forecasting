"""在项目根目录运行 python -m E.experiment train 或 test。"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from C.models import TCN, baselines
from D.models import STGCN, make_graph

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "E/settings.json").read_text(encoding="utf-8-sig"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUT1, CUT2, END = 731*24, 1096*24, 1461*24
H = CFG["horizons"]
CACHE = ROOT / "cache"
OUT = ROOT / "E/results"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
torch.set_float32_matmul_precision("high")


def stats(x):
    # 分块计算保证预报数组的统计不会占满内存。
    count = np.zeros(x.shape[-1], dtype=np.float64)
    total, square = count.copy(), count.copy()
    for i in range(0, CUT1-H, 96):
        a = x[i:min(i+96, CUT1-H)]
        mask = np.isfinite(a)
        flat = np.where(mask, a, 0).reshape(-1, a.shape[-1]).astype(np.float64)
        count += mask.reshape(flat.shape).sum(0)
        total += flat.sum(0)
        square += np.einsum("ij,ij->j",flat,flat)
    mean = total / np.maximum(count,1)
    sd = np.sqrt(np.maximum(square/np.maximum(count,1)-mean**2,1.e-6))
    return mean.astype("f"), sd.astype("f")


def encode(raw, mean, sd):
    z = (raw-mean)/sd
    finite = np.isfinite(z)
    return np.concatenate([np.where(finite, np.clip(z,-8,8), 0), finite.astype("f")],axis=-1)


class Data:
    def __init__(self, weather):
        self.stations = pd.read_csv(CACHE/"A/stations.csv",dtype={"station_id":str,"city_id":str,"district_id":str})
        self.air = np.load(CACHE/"A/air.npy",mmap_mode="r")
        mean, sd = stats(self.air)
        self.center, self.scale = float(mean[0]), float(sd[0])
        self.air_x = torch.as_tensor(encode(self.air,mean,sd),device=DEVICE)
        self.y = torch.tensor(self.air[:,:,0].copy(),device=DEVICE)
        self.ago = torch.arange(-47,1,device=DEVICE)
        self.ahead = torch.arange(1,25,device=DEVICE)
        self.met = self.fc = None
        if weather:
            met=np.load(CACHE/"B/met.npy",mmap_mode="r")
            self.met=torch.as_tensor(encode(met,*stats(met)),device=DEVICE)
            self.fc=np.load(CACHE/"B/forecast.npy",mmap_mode="r")
            self.fc_stats=stats(self.fc)

    def batch(self, origins, hist_met, use_fc):
        t = torch.as_tensor(origins,device=DEVICE)
        past = t[:,None]+self.ago
        x = self.air_x[past]
        if hist_met:
            x=torch.cat([x,self.met[past]],-1)
        future=None
        if use_fc:
            future=torch.as_tensor(encode(self.fc[origins],*self.fc_stats),device=DEVICE)
        y=self.y[t[:,None]+self.ahead].transpose(1,2)
        # 以最近 48 小时的有效 PM2.5 为多步回归的共同起点。
        valid=x[...,6]>0
        ix=(valid*torch.arange(1,49,device=DEVICE)[None,:,None]).amax(1)
        anchor=x[...,0].gather(1,(ix-1).clamp_min(0).unsqueeze(1)).squeeze(1)
        anchor=torch.where(ix>0,anchor,0)
        return x,future,y,anchor


def experiment_model(data, exp, k):
    kind, met, fc, loops = CFG["experiments"][exp]
    channels=data.air_x.shape[-1]+(data.met.shape[-1] if met else 0)
    future=data.fc.shape[-1]*2 if fc else 0
    if kind=="TCN":
        return TCN(channels,future_channels=future).to(DEVICE)
    adjacency=make_graph(data.stations,k=k,self_only=loops)
    return STGCN(channels,adjacency,future_channels=future).to(DEVICE)


def predict(model,data,exp,origins):
    _,met,fc,_=CFG["experiments"][exp]
    x,f,y,anchor=data.batch(origins,met,fc)
    return (model(x,f)+anchor.unsqueeze(-1))*data.scale+data.center,y


def batches(indices):
    for i in range(0,len(indices),CFG["batch"]):
        yield indices[i:i+CFG["batch"]]


def validate(model,data,exp):
    model.eval()
    total,count=0.,0
    with torch.no_grad():
        for ix in batches(np.arange(CUT1-1,CUT2-H,CFG["validation_origin_stride"])):
            pred,y=predict(model,data,exp,ix)
            ok=torch.isfinite(y)
            total+=torch.where(ok,(pred-y).abs(),0).sum().item()
            count+=ok.sum().item()
    return total/count


def fit(data,exp,seed,lr,k,trial):
    tag=f"{exp}_{seed}_{trial}"
    role="C" if exp=="M1" else "D"
    directory=ROOT/role/"runs"
    directory.mkdir(exist_ok=True)
    log=directory/(tag+".json")
    weights=directory/(tag+".pt")
    if log.exists() and weights.exists():
        return json.loads(log.read_text())
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng=np.random.default_rng(seed)
    model=experiment_model(data,exp,k)
    optim=torch.optim.AdamW(model.parameters(),lr=lr)
    curve=[]
    start=time.perf_counter()
    for epoch in range(CFG["epochs"]):
        model.train()
        stride=CFG["train_origin_stride"]
        ix=np.arange(47+(seed+epoch*5)%stride,CUT1-H,stride)
        rng.shuffle(ix)
        for part in batches(ix):
            pred,y=predict(model,data,exp,part)
            valid=torch.isfinite(y)
            target=torch.nan_to_num(y,nan=data.center)
            loss=torch.where(valid,(pred-target).abs(),0).sum()/valid.sum().clamp_min(1)/data.scale
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
            optim.step()
        score=validate(model,data,exp)
        if not np.isfinite(score):
            raise ValueError(f"{tag} validation MAE={score}")
        if not curve or score<min(curve):
            torch.save(model.state_dict(),weights)
        curve.append(score)
    result=dict(tag=tag,role=role,experiment=exp,seed=seed,lr=lr,k=k,validation_mae=min(curve),
                epoch=int(np.argmin(curve))+1,curve=curve,parameters=sum(p.numel() for p in model.parameters()),
                seconds=round(time.perf_counter()-start,2))
    log.write_text(json.dumps(result,indent=2))
    print(tag, result["validation_mae"],result["seconds"],flush=True)
    return result


def train(experiments):
    data=Data(any(CFG["experiments"][e][1] for e in experiments))
    for exp in experiments:
        # 固定同样的训练预算。在基础模型验证集上选参，消融沿用 STGCN 配置。
        if exp in ["M1","M2"]:
            trials=[fit(data,exp,17,lr,k,i) for i,(lr,k) in enumerate(zip(CFG["learning_rates"],CFG["neighbors"]))]
            first=min(trials,key=lambda r:r["validation_mae"])
        else:
            base=json.loads((OUT/"frozen_M2.json").read_text())[0]
            first=fit(data,exp,17,base["lr"],base["k"],0)
        runs=[first]+[fit(data,exp,seed,first["lr"],first["k"],0) for seed in CFG["seeds"][1:]]
        (OUT/f"frozen_{exp}.json").write_text(json.dumps(runs,indent=2))


def test():
    frozen={e:json.loads((OUT/f"frozen_{e}.json").read_text()) for e in CFG["experiments"]}
    data=Data(True)
    origins=np.arange(CUT2-1,END-H)
    folder=CACHE/"predictions"
    folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/"origins.npy",origins)
    shape=(len(origins),len(data.stations),H)
    def allocate(name):
        return np.lib.format.open_memmap(folder/(name+".npy"),mode="w+",shape=shape,dtype="float32")
    y,p1,p2=[allocate(n) for n in ["truth","B1","B2"]]
    for ix in batches(np.arange(len(origins))):
        t=origins[ix]
        past=data.air[t[:,None]+np.arange(-47,1),:,0]
        y[ix]=data.air[t[:,None]+np.arange(1,25),:,0].transpose(0,2,1)
        p1[ix],p2[ix]=baselines(past)
    for exp,runs in frozen.items():
        for r in runs:
            model=experiment_model(data,exp,r["k"])
            model.load_state_dict(torch.load(ROOT/r["role"]/"runs"/(r["tag"]+".pt"),map_location=DEVICE,weights_only=True))
            model.eval()
            dest=allocate(f"{exp}_{r['seed']}")
            with torch.no_grad():
                for ix in batches(np.arange(len(origins))):
                    pred,_=predict(model,data,exp,origins[ix])
                    dest[ix]=pred.cpu().numpy()
            dest.flush()
            print("tested",exp,r["seed"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["train","test"])
    p.add_argument("experiments",nargs="*")
    a=p.parse_args()
    if a.action=="train": train(a.experiments or list(CFG["experiments"]))
    else: test()

