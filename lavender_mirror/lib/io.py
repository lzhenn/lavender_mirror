#!/usr/bin/env python3
"""specific module for IO"""
# ---imports---
import os, datetime
import pandas as pd
import xarray as xr
import numpy as np
from . import utils
import wrf
import netCDF4 as nc4
# ---Module regime consts and variables---
print_prefix='lib.io>>'


# ---Classes and Functions---
def read_cruise(cfg):
    '''
    read cruise data
    '''
    dfs=[]
    cruise_dir=cfg['INPUT']['cruise_dir'] 
    # Loop through all files in the directory
    for filename in os.listdir(cruise_dir):
        if filename.endswith('.txt'):
            utils.write_log(f'{print_prefix}Reading {filename}...')            
            # Read the CSV file with ';' delimiter
            df = pd.read_csv(os.path.join(cruise_dir, filename), delimiter=';',
                header=None, usecols=[0, 1, 2, 3], 
                names=['time', 'lon', 'lat', 'tvoc'], parse_dates=['time'])
            # Add the dataframe to the list
            dfs.append(df)
    
    # Concatenate all dataframes in the list
    res_df = pd.concat(dfs, ignore_index=True)
    res_df = res_df.resample('1T', on='time').mean()
    res_df = res_df.dropna(subset=['tvoc'])
    return res_df

def read_episode(cfg):
    '''
    read episode data
    '''
    episode_fn=cfg['INPUT']['episode_file'] 
    rsmp_frq=cfg['INPUT']['resample_frq']+'S'
    tgt=cfg['INPUT']['tgt_species'] 
    try:
        # Standard format
        df = pd.read_csv(episode_fn)
        df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d_%H:%M:%S')
 
    except:
        # Read the CSV file with ';' delimiter
        df = pd.read_csv(episode_fn, delimiter=';',
                header=None, usecols=[0, 1, 2, 3], 
                names=['time', 'lon', 'lat', 'TVOC2'], parse_dates=['time'])
 
    if cfg['INPUT'].getboolean('parse_wind'):
        try:
            wspd,wdir=df['wspd'].values,df['wdir'].values
        except:
            utils.throw_error(f'{print_prefix}wspd or wdir parser error in {episode_fn}!')
        u1d,v1d=np.zeros(len(wspd)),np.zeros(len(wspd))
        for id in range(len(wspd)):
            u1d[id],v1d[id]=utils.wswd2uv(wspd[id],wdir[id])
        df['u'],df['v']=u1d,v1d
    res_df = df.resample(rsmp_frq, on='time').mean()
    res_df = res_df.dropna(subset=[tgt])
    return(res_df)

def feed_uv(cfg, tfs):
    interval=int(cfg['INPUT']['feed_frq'])
    wrfout_path=cfg['INPUT']['wrfout_path']
    wrf_tfs = tfs[tfs.strftime('%M').isin(
        [str(i).zfill(2) for i in range(0, 60, interval)])]
    # to UTC
    wrf_tfs=wrf_tfs-datetime.timedelta(hours=8)
    wrflst=[os.path.join(
        wrfout_path,tf.strftime('wrfout_d04_%Y-%m-%d_%H:%M:%S')) for tf in wrf_tfs]
    wrfouts=[]
    for fn in wrflst:
        if not os.path.exists(fn):
            utils.write_log(f'{print_prefix}File {fn} does not exist!') 
            exit()
        wrfouts.append(nc4.Dataset(fn))
    XLAT=wrf.getvar(wrfouts[0],'XLAT')
    XLONG=wrf.getvar(wrfouts[0],'XLONG')
    U10 = wrf.getvar(wrfouts, 'U10',timeidx=wrf.ALL_TIMES, method="join")
    V10 = wrf.getvar(wrfouts, 'V10',timeidx=wrf.ALL_TIMES, method="join")
    return XLAT, XLONG, U10, V10, wrf_tfs
def outnc(cfg, lat2d, lon2d, vardic):
    outpath=cfg['OUTPUT']['nc_path']
    lat1d,lon1d=lat2d[:,0],lon2d[0,:]
    dadic={}
    for key, val in vardic.items():
        dadic[key]= xr.DataArray(val, dims=('lat', 'lon'), 
            coords={'lat': lat1d, 'lon': lon1d})
    # create a Dataset
    ds = xr.Dataset(dadic)
    # write the Dataset to a netCDF file
    ds.to_netcdf(outpath)
    
# ---Unit test---
if __name__ == '__main__':
    pass
