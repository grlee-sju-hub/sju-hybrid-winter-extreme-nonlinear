#!/usr/bin/env python3
"""ERA5 ACC corr-leads: Korea extreme index (korea_nday for EC/EW/HW) vs ERA5 monthly fields, leads 0-3,
detrended Pearson per grid. L0 = season mean (DJF=Dec+Jan+Feb, JA=Jul+Aug); L_k = k months before season start
(DJF: Nov/Oct/Sep ; JA: Jun/May/Apr). Out: data/corr/{EC,EW,HW}_era5_corr.nc  r/p(var,lead,lat,lon)."""
import glob, numpy as np, xarray as xr
from pathlib import Path
from scipy import stats
P=Path("."); MON=P/"data"/"ERA5"/"monthly"; CORR=P/"data"/"corr"; CORR.mkdir(exist_ok=True)
VARS=["t2m","msl","sst","siconc","z500","u200","v200","t850","weasd","liqvsm","tprate","z850"]
IDX=[("EC","DJF","DJF_EC"),("EW","DJF","DJF_EW"),("HW","JA","JA_HW")]; LEADS=[0,1,2,3]
def lead_months(season,lead,Y):
    if season=="DJF": return [(Y,12),(Y+1,1),(Y+1,2)] if lead==0 else [(Y,{1:11,2:10,3:9}[lead])]
    return [(Y,7),(Y,8)] if lead==0 else [(Y,{1:6,2:5,3:4}[lead])]
print("loading ERA5 monthly store ...",flush=True)
monthly={}; LAT=LON=None
for f in sorted(glob.glob(str(MON/"ERA5m_*.nc"))):
    ym=Path(f).stem.split("_")[1]; Y,Mo=int(ym[:4]),int(ym[4:])
    d=xr.open_dataset(f); monthly[(Y,Mo)]={v:d[v].values.astype("float32") for v in VARS}
    LAT=d.latitude.values; LON=d.longitude.values; d.close()
nla,nlo=len(LAT),len(LON); print("loaded %d months  grid %dx%d"%(len(monthly),nla,nlo),flush=True)
def detr1(x):
    x=np.asarray(x,float); n=len(x); m=np.isfinite(x); c=np.polyfit(np.arange(n)[m],x[m],1); return x-(c[0]*np.arange(n)+c[1])
def corr_map(idxv,field):
    yrs=np.where(np.isfinite(idxv))[0]; iv=detr1(idxv[yrs]); iva=iv-iv.mean(); nn=len(yrs)
    F=field[yrs]; valid=np.isfinite(F).all(0)
    r=np.full((nla,nlo),np.nan,"float32"); p=np.full((nla,nlo),np.nan,"float32")
    if valid.sum()==0: return r,p
    Fv=F[:,valid]; tt=np.arange(nn,dtype=float); A=np.vstack([tt,np.ones(nn)]).T
    coef,_,_,_=np.linalg.lstsq(A,Fv,rcond=None); Fd=Fv-A@coef; Fda=Fd-Fd.mean(0)
    num=(iva[:,None]*Fda).sum(0); den=np.sqrt((iva**2).sum()*(Fda**2).sum(0)); rv=np.where(den>0,num/den,np.nan)
    with np.errstate(invalid="ignore",divide="ignore"):
        ts=rv*np.sqrt((nn-2)/(1-rv**2)); pv=2*stats.t.sf(np.abs(ts),nn-2)
    r[valid]=rv; p[valid]=pv.astype("float32"); return r,p
for name,season,fn in IDX:
    d=xr.open_dataset(P/"data"/f"ERA5_{fn}_index_1979-2024.nc"); yrs=d.year.values.astype(int); pred=d.korea_nday.values.astype(float); d.close()
    R=np.full((len(VARS),len(LEADS),nla,nlo),np.nan,"float32"); PP=R.copy()
    for li,lead in enumerate(LEADS):
        for vi,var in enumerate(VARS):
            field=np.full((len(yrs),nla,nlo),np.nan,"float32")
            for yi,Y in enumerate(yrs):
                ms=lead_months(season,lead,int(Y)); arrs=[monthly[k][var] for k in ms if k in monthly]
                if len(arrs)==len(ms): field[yi]=np.mean(arrs,0)
            R[vi,li],PP[vi,li]=corr_map(pred,field)
    xr.Dataset({"r":(("var","lead","lat","lon"),R),"p":(("var","lead","lat","lon"),PP)},
        coords={"var":VARS,"lead":LEADS,"lat":LAT,"lon":LON},
        attrs={"index":name,"season":season,"predictand":"ERA5 korea_nday","period":"1979-2024",
               "method":"detrended Pearson; L0=season mean, L_k=k months before season start; ERA5 monthly fields"}
        ).to_netcdf(CORR/f"{name}_era5_corr.nc")
    print("[saved] %s_era5_corr.nc"%name,flush=True)
print("CORR_DONE",flush=True)
