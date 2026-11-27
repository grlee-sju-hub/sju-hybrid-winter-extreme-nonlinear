#!/usr/bin/env python3 -I
# EC/EW tercile — CatBoost. BOTH 8-model count (N=8) & full-ensemble (N=219, pooled calib). Estimator=CatBoost LOOCV. Hits only.
# Lead predictors use observed value; only L0 carry member spread. usage: INDEX = EC_day|EW_day
import sys, numpy as np, pandas as pd, xarray as xr
from pathlib import Path
from catboost import CatBoostRegressor
import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["font.family"]="Nimbus Sans"; matplotlib.rcParams["font.size"]=12; matplotlib.rcParams["mathtext.fontset"]="stixsans"
import matplotlib.pyplot as plt
D=Path("data"); FIG=Path("figure")
IDX=sys.argv[1] if len(sys.argv)>1 else "EC_day"; PH={"EC_day":"EC","EW_day":"EW"}[IDX]
MODELS={"EC_day":[10,191,132],"EW_day":[218,313,265]}; gids=MODELS[IDX]
COL=["#36b5ff","#d9d9d9","#ff5a4e"]; EV=[int(y) for y in range(1993,2017)]; Y46=np.arange(1979,2025); pos={int(Y):k for k,Y in enumerate(Y46)}
CB=dict(iterations=200,depth=2,learning_rate=0.05,l2_leaf_reg=5.0,loss_function="RMSE",random_seed=42,verbose=0,allow_writing_files=False)
def detr(x): x=np.asarray(x,float); n=len(x); t=np.arange(n)-(n-1)/2.; a=x-x.mean(); s=(t*t).sum(); return a-((t*a).sum()/s)*t
d=xr.open_dataset(D/f"ERA5_DJF_{PH}_index_1979-2024.nc"); yq=d.year.values.astype(int); ym={int(a):float(b) for a,b in zip(yq,d.korea_nday.values.astype(float))}; d.close()
dz=np.load(D/"predictor_accepted_era5.npz",allow_pickle=True); G=dz["gid"].astype(int); OBS=dz["obs_ts"]; LEAD=dz["lead"].astype(int); gi={int(g):i for i,g in enumerate(G)}
L0=[g for g in gids if LEAD[gi[g]]==0]
yt=detr(np.array([ym[Y] for Y in Y46])); Xf={g:detr(OBS[gi[g]][[pos[Y] for Y in Y46]]) for g in gids}
OM={g:np.array([Xf[g][pos[Y]] for Y in EV]).mean() for g in L0}; OSD={g:np.array([Xf[g][pos[Y]] for Y in EV]).std() for g in L0}
pm=pd.read_csv(D/"c3s_ensemble_L0_djf.csv"); pm=pm[pm.gid.isin(L0)]; models=sorted(pm.model.unique())
MOD={}
for Y in EV:
    tr=Y46!=Y; MOD[Y]=CatBoostRegressor(**CB); MOD[Y].fit(np.column_stack([Xf[g][tr] for g in gids]),yt[tr])
def calib(mode):
    cal={g:{} for g in L0}
    for g in L0:
        for mo in models:
            sub=pm[(pm.gid==g)&(pm.model==mo)]
            if mode=="8model":
                gm=sub.groupby("year")["value"].mean(); cm,cs=gm.mean(),gm.std()
                for Y in gm.index: cal[g][(mo,int(Y),None)]=(gm.loc[Y]-cm)/cs*OSD[g]+OM[g]
            else:
                cm,cs=sub["value"].mean(),sub["value"].std()
                for _,r in sub.iterrows(): cal[g][(mo,int(r.year),int(r.member))]=(r["value"]-cm)/cs*OSD[g]+OM[g]
    return cal
def votes(mode):
    cal=calib(mode); out={Y:[] for Y in EV}
    for mo in models:
        sub=pm[pm.model==mo]
        if mode=="8model":
            for Y in EV:
                if all((mo,int(Y),None) in cal[g] for g in L0):
                    out[Y].append([(cal[g][(mo,int(Y),None)] if g in L0 else Xf[g][pos[Y]]) for g in gids])
        else:
            my=sub.groupby("year")["member"].apply(lambda s:sorted(set(int(x) for x in s)))
            for Y in EV:
                if Y not in my.index: continue
                for m in my.loc[Y]:
                    if all((mo,int(Y),m) in cal[g] for g in L0):
                        out[Y].append([(cal[g][(mo,int(Y),m)] if g in L0 else Xf[g][pos[Y]]) for g in gids])
    return out
t33,t67=np.percentile(yt,[100/3,200/3])
def catf(v): return 0 if v<t33 else (2 if v>t67 else 1)
def evaluate(mode):
    vv=votes(mode); frac=np.zeros((len(EV),3)); ocat=[]; nv=[]
    for i,Y in enumerate(EV):
        preds=[float(MOD[Y].predict(np.array([x],dtype=float))[0]) for x in vv[Y]]; cs=[catf(p) for p in preds]; n=max(len(cs),1); nv.append(len(cs))
        for c in cs: frac[i,c]+=1.0/n
        ocat.append(catf(yt[pos[Y]]))
    hit=0.0; hm=[]
    for i in range(len(EV)):
        mx=frac[i].max(); win=[c for c in range(3) if abs(frac[i,c]-mx)<1e-9]
        if ocat[i] in win: hit+=1.0/len(win); hm.append("check" if len(win)==1 else "half")
        else: hm.append(None)
    return frac,ocat,hit,hm,int(np.median(nv))
def draw(frac,ocat,hit,hm,title,out):
    fig,ax=plt.subplots(figsize=(6.56,3.2)); P=frac*100; bottom=np.zeros(len(EV))
    for c in range(3):
        ax.bar(EV,P[:,c],bottom=bottom,color=COL[c],width=0.84,edgecolor="white",lw=0.4,zorder=2)
        for i in range(len(EV)):
            if P[i,c]>=7: ax.text(EV[i],bottom[i]+P[i,c]/2,f"{P[i,c]:.0f}",ha="center",va="center",fontsize=8,color=("black" if c==1 else "white"),zorder=4)
        bottom=bottom+P[:,c]
    OBSM={0:("v","#1f78d6"),1:("o","#888888"),2:("^","#d62728")}
    for i in range(len(EV)):
        m,cc=OBSM[ocat[i]]; ax.scatter([EV[i]],[105],marker=m,s=34,c=cc,clip_on=False,zorder=5)
        if hm[i]=="check": ax.text(EV[i],113,"✓",ha="center",va="bottom",fontsize=12,color="#239a3b",fontweight="bold",fontfamily="DejaVu Sans",clip_on=False,zorder=5)
        elif hm[i]=="half": ax.text(EV[i],113,"½",ha="center",va="bottom",fontsize=11,color="#555",fontfamily="DejaVu Sans",clip_on=False,zorder=5)
    ax.set_ylim(0,100); ax.set_xlim(EV[0]-0.7,EV[-1]+0.7); ax.set_yticks([0,33.3,66.7,100]); ax.set_yticklabels(["0","33","67","100"]); ax.set_ylabel("Probability (%)")
    xt=[Y for Y in EV if Y%2==0]; ax.set_xticks(xt); ax.set_xticklabels([f"{Y%100:02d}" for Y in xt])
    ax.text(0.0,1.30,title,transform=ax.transAxes,ha="left",va="bottom",fontsize=13.5)
    ax.text(1.0,1.30,f"Hits {hit:.1f}/{len(EV)}",transform=ax.transAxes,ha="right",va="bottom",fontsize=12.5)
    fig.subplots_adjust(left=0.115,right=0.985,bottom=0.13,top=0.72)
    fig.savefig(out,dpi=200,transparent=True); plt.close(fig); print("[saved]",out)
for mode,fname in [("8model",f"fig_tercile_count_{IDX}_cb.png"),("member",f"fig_tercile_member_{IDX}_cb.png")]:
    fr,oc,ht,hm,nv=evaluate(mode); print("[%s] catboost %-7s Hits %.1f/%d votes~%d"%(IDX,mode,ht,len(EV),nv))
    draw(fr,oc,ht,hm,f"DJF {PH} Days Tercile Forecast  (CatBoost, N={nv})",FIG/fname)
