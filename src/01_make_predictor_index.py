#!/usr/bin/env python3
"""ERA5 predictor indexization + 1st screening, on C3S grid (180x360).
Masks read from precomputed ERA5_{idx}_corr_c3s180 (Spearman rS/pS) & C3S_skill_{seas}_c3s180 (ensures figure consistency);
area-mean series from ERA5 monthly regridded (2x2 midpoint) to C3S. L0 mask = Spearman-sig & skill-sig & skill r>0 ;
lead L1-3 mask = Spearman-sig only (no skill; observed by fc time). Periodic-8 clusters >=75 -> cos-lat area-mean ->
target_r=Spearman(detrend region, detrend predictand) -> greedy collinearity(|rho|>0.6)+SH excl(lat_max<=-30).
Out: data/predictor_region_meta_era5.csv, data/predictor_accepted_era5.npz, data/predictor_index_era5.xlsx.
Usage: [IDX] -> only that predictand (test)."""
import sys, glob, numpy as np, pandas as pd, xarray as xr
import scipy.ndimage as ndi
from scipy.stats import rankdata, spearmanr, t as tdist
from pathlib import Path
P=Path("."); D=P/"data"; MON=D/"ERA5"/"monthly"
MIN_CELLS=100; R_DROP=0.6; LAT_MAX_S=-30.0; YR0,YR1=1979,2024; years=np.arange(YR0,YR1+1)   # 100 cells @1deg (C3S grid)
VARS=["t2m","msl","sst","siconc","tprate","z500","u200","v200","t850","z850"]  # L0 skill-gated (C3S 10, z850 added)
VARS_LEAD=VARS+["weasd","liqvsm"]                                              # lead (no skill): +snow/soilm = 12
LEADS=[0,1,2,3]; LEADTOK={"DJF":{0:"DJF",1:"NOV",2:"OCT",3:"SEP"},"JA":{0:"JA",1:"JUN",2:"MAY",3:"APR"}}
TOKMON={"DJF":[(0,12),(1,1),(1,2)],"NOV":[(0,11)],"OCT":[(0,10)],"SEP":[(0,9)],
        "JA":[(0,7),(0,8)],"JUN":[(0,6)],"MAY":[(0,5)],"APR":[(0,4)]}
IDX=[("EC_day","EC","DJF"),("EW_day","EW","DJF"),("CDHD","CDHD","JA")]   # summer = compound CDHD (not HWD/DR separately)
CLAT=np.arange(89.5,-90.0,-1.0); CLON=np.arange(0.5,360.0,1.0); nla,nlo=len(CLAT),len(CLON)
def regrid_e2c(a): la=0.5*(a[...,:-1,:]+a[...,1:,:]); return 0.5*(la+np.roll(la,-1,axis=-1))
def detr1(x):
    x=np.asarray(x,float); n=len(x); mm=np.isfinite(x)
    if mm.sum()<4: return x-np.nanmean(x)
    c=np.polyfit(np.arange(n)[mm],x[mm],1); return x-(c[0]*np.arange(n)+c[1])
def pred_series(name):
    if name=="EC_day": d=xr.open_dataset(D/"ERA5_DJF_EC_index_1979-2024.nc"); y=d.year.values.astype(int); v=d.korea_nday.values.astype(float); d.close()
    elif name=="EW_day": d=xr.open_dataset(D/"ERA5_DJF_EW_index_1979-2024.nc"); y=d.year.values.astype(int); v=d.korea_nday.values.astype(float); d.close()
    elif name=="CDHD":                                    # compound hot-dry index (precomputed Korea-mean series)
        d=xr.open_dataset(D/"ERA5_JA_CDHD_index_1979-2024.nc"); y=d.year.values.astype(int); v=d.korea_nday.values.astype(float); d.close()
    ymap={int(a):b for a,b in zip(y,v)}; return np.array([ymap.get(int(Y),np.nan) for Y in years])
ONLY=sys.argv[1] if len(sys.argv)>1 else None
idxlist=[x for x in IDX if (ONLY is None or x[0]==ONLY)]
# ---- build ERA5 field aggregates on C3S grid (for area-mean) ----
print("loading+regridding ERA5 monthly -> C3S, building aggregates ...",flush=True)
monthly={}
for f in sorted(glob.glob(str(MON/"ERA5m_*.nc"))):
    ym=Path(f).stem.split("_")[1]; Y,Mo=int(ym[:4]),int(ym[4:])
    d=xr.open_dataset(f); monthly[(Y,Mo)]={v:regrid_e2c(d[v].values.astype("float32")) for v in VARS_LEAD}; d.close()
seasons_used=set(s for _,_,s in idxlist); toks_used=set()
for s in seasons_used: toks_used|=set(LEADTOK[s].values())
AGG={}
for v in VARS_LEAD:
    for tok in toks_used:
        arr=np.full((len(years),nla,nlo),np.nan,"float32")
        for i,Y in enumerate(years):
            fs=[monthly[(Y+yo,mo)][v] for (yo,mo) in TOKMON[tok] if (Y+yo,mo) in monthly]
            if len(fs)==len(TOKMON[tok]): arr[i]=np.mean(fs,0)
        AGG[(v,tok)]=arr
del monthly; print("aggregates built: %d (v,tok)"%len(AGG),flush=True)
w2d=np.cos(np.deg2rad(CLAT))[:,None]*np.ones((1,nlo)); lat2d=CLAT[:,None]*np.ones((1,nlo)); lon2d=np.ones((nla,1))*CLON[None,:]
def label_periodic(binary):
    ny,nx=binary.shape; ext=np.concatenate([binary,binary],axis=1)
    lab,_=ndi.label(ext,structure=np.ones((3,3),int)); left,right=lab[:,:nx],lab[:,nx:]
    act=left>0; pairs={(int(a),int(b)) for a,b in zip(left[act].tolist(),right[act].tolist()) if a>0 and b>0 and a!=b}
    parent={}
    def find(x):
        while parent.get(x,x)!=x: parent[x]=parent.get(parent[x],parent[x]); x=parent[x]
        return x
    for a,b in pairs:
        ra,rb=find(a),find(b)
        if ra!=rb: parent[ra]=rb
    out=np.zeros((ny,nx),np.int32)
    for u in np.unique(left):
        if u==0: continue
        out[left==u]=find(int(u))
    return out
def area_mean(field,rmask):
    w=w2d[rmask]; vals=field[:,rmask]; fin=np.isfinite(vals); we=np.broadcast_to(w,vals.shape)
    num=np.where(fin,vals*we,0).sum(1); den=np.where(fin,we,0).sum(1); return np.where(den>0,num/den,np.nan)
meta=[]; obs_ts_all={}; mask_all={}; gid=0
for name,cnm,season in idxlist:
    pred=pred_series(name)
    dc=xr.open_dataset(D/f"ERA5_{cnm}_corr_c3s180_1979-2024.nc"); cvar=list(dc["var"].values)
    sk=xr.open_dataset(D/f"C3S_skill_{season}_c3s180_1993-2016.nc"); svar=list(sk["var"].values)
    for lead in LEADS:
        tok=LEADTOK[season][lead]
        for v in (VARS if lead==0 else VARS_LEAD):
            rS=dc.rS.isel(var=cvar.index(v),lead=lead).values; pS=dc.pS.isel(var=cvar.index(v),lead=lead).values
            if lead==0:
                rs=sk.r.isel(var=svar.index(v)).values; ps=sk.p.isel(var=svar.index(v)).values
                m=(pS<0.05)&(ps<0.05)&(rs>0)&np.isfinite(rS)
            else:
                m=(pS<0.05)&np.isfinite(rS)
            gate=np.where(m&(rS>0),1,np.where(m&(rS<0),-1,0))
            field=AGG[(v,tok)]
            for sign in (1,-1):
                lab=label_periodic(gate==sign)
                if lab.max()==0: continue
                for uid in np.unique(lab[lab>0]):
                    rmask=(lab==uid); ncl=int(rmask.sum())
                    if ncl<MIN_CELLS: continue
                    gid+=1; obs_ts_all[gid]=area_mean(field,rmask).astype("float32"); mask_all[gid]=rmask
                    lr=np.deg2rad(lon2d[rmask]); lonm=(np.rad2deg(np.arctan2(np.sin(lr).mean(),np.cos(lr).mean()))+360)%360
                    meta.append(dict(gid=gid,index=name,var=v,lead=lead,sign=sign,ncells=ncl,
                                     lat_mean=round(float(lat2d[rmask].mean()),1),lon_mean=round(float(lonm),1),
                                     lat_max=float(lat2d[rmask].max())))
    dc.close(); sk.close()
meta=pd.DataFrame(meta)
print("candidate regions: %d  (by lead: %s)"%(len(meta),meta.groupby('lead').size().to_dict() if len(meta) else {}),flush=True)
def sp(a,b):
    mm=np.isfinite(a)&np.isfinite(b)
    if mm.sum()<10: return np.nan,np.nan
    r,p=spearmanr(a[mm],b[mm]); return float(r),float(p)
obs_dt={g:detr1(obs_ts_all[g]) for g in obs_ts_all}
meta["target_r"]=np.nan; meta["target_p"]=np.nan
for name,cnm,season in idxlist:
    pred=detr1(pred_series(name))
    for i in meta[meta["index"]==name].index:
        r,p=sp(obs_dt[meta.at[i,"gid"]],pred); meta.at[i,"target_r"]=r; meta.at[i,"target_p"]=p
def greedy(R,annotate=False):
    if annotate: meta["accepted"]=False; meta["rank_screen"]=0; meta["dropped_by"]=0
    counts={}
    for name,_,_ in idxlist:
        sub=meta[meta["index"]==name]
        if not len(sub): counts[name]=0; continue
        order=meta.loc[sub.index].reindex(meta.loc[sub.index,"target_r"].abs().sort_values(ascending=False).index).index
        acc=[]
        for i in order:
            if meta.at[i,"lat_max"]<=LAT_MAX_S:
                if annotate: meta.at[i,"dropped_by"]=-2
                continue
            if not np.isfinite(meta.at[i,"target_r"]):
                if annotate: meta.at[i,"dropped_by"]=-1
                continue
            drop=False
            for j in acc:
                rr,_=sp(obs_dt[meta.at[i,"gid"]],obs_dt[meta.at[j,"gid"]])
                if np.isfinite(rr) and abs(rr)>R:
                    if annotate: meta.at[i,"dropped_by"]=int(meta.at[j,"gid"])
                    drop=True; break
            if not drop:
                if annotate: meta.at[i,"accepted"]=True; meta.at[i,"rank_screen"]=len(acc)+1
                acc.append(i)
        counts[name]=len(acc)
    return counts
print("R_DROP sensitivity:",flush=True)
for R in (0.5,0.6,0.7): c=greedy(R); print("  R=%s total %d %s"%(R,sum(c.values()),c),flush=True)
greedy(R_DROP,annotate=True)
meta.to_csv(D/"predictor_region_meta_era5.csv",index=False)
accdf=meta[meta["accepted"]].sort_values(["index","rank_screen"]); gids=accdf["gid"].values
np.savez_compressed(D/"predictor_accepted_era5.npz",
    gid=gids,index=accdf["index"].values.astype(str),var=accdf["var"].values.astype(str),
    lead=accdf["lead"].values.astype(int),sign=accdf["sign"].values.astype(int),
    target_r=accdf["target_r"].values.astype(float),lat_mean=accdf["lat_mean"].values.astype(float),
    lon_mean=accdf["lon_mean"].values.astype(float),
    obs_ts=np.stack([obs_ts_all[g] for g in gids]).astype("float32") if len(gids) else np.zeros((0,len(years)),"float32"),
    mask=np.stack([mask_all[g].astype("uint8") for g in gids]) if len(gids) else np.zeros((0,nla,nlo),"uint8"),
    lat=CLAT,lon=CLON,years=years)
acc=meta[meta["accepted"]].copy()
print("[saved] predictor_accepted_era5.npz (%d series)"%len(gids),flush=True)
print("ACCEPTED %d  by index: %s"%(len(acc),acc.groupby('index').size().to_dict()),flush=True)
print("by index x lead:\n"+acc.groupby(["index","lead"]).size().to_string(),flush=True)
cols=["index","var","lead","sign","ncells","lat_mean","lon_mean","target_r","target_p","rank_screen"]
with pd.ExcelWriter(D/"predictor_index_era5.xlsx") as xw:
    acc[cols].sort_values(["index","lead","rank_screen"]).round(3).to_excel(xw,sheet_name="accepted",index=False)
    meta.round(3).to_excel(xw,sheet_name="all_candidates",index=False)
print("[saved] predictor_index_era5.xlsx + region_meta_era5.csv",flush=True); print("INDEX_DONE",flush=True)
