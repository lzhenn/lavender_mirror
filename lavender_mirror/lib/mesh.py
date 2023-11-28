#!/usr/bin/env python3
import numpy as np
import pandas as pd
import datetime
import xarray as xr
from . import io, utils
from scipy.spatial.distance import cdist

print_prefix='lib.mesh>>'
def build_meshgrid(cfg):
    # Define SW and NE points
    # sw_lat, sw_lon = 30.768639, 121.418872 
    # ne_lat, ne_lon = 30.823493, 121.482215
    # sw_lat, sw_lon = 31.14, 121.23 
    # ne_lat, ne_lon = 31.4, 121.7
    sw_lat = float(cfg['INPUT']['lon_min'])
    sw_lon = float(cfg['INPUT']['lat_min'])
    ne_lat = float(cfg['INPUT']['lon_max'])
    ne_lon = float(cfg['INPUT']['lat_max'])
    
    #res=0.001 # 0.001 ~ 100m
    res=0.00001*int(cfg['KERNEL']['res_mesh'])

    # Calculate number of points in latitude and longitude direction
    lat_points = int((ne_lat - sw_lat) / res) + 1
    lon_points = int((ne_lon - sw_lon) / res) + 1

    # Create latitude and longitude arrays
    lat_array = np.linspace(sw_lat, ne_lat, lat_points)
    lon_array = np.linspace(sw_lon, ne_lon, lon_points)

    # Create latitude and longitude meshgrid
    lon_mesh, lat_mesh = np.meshgrid(lon_array, lat_array)
    return lon_mesh, lat_mesh
def build_uvmesh(cfg, df, lat_mesh, lon_mesh):
    df_tfs=df.index
    #tfs=pd.date_range(
    #    df_tfs[0]-datetime.timedelta(hours=1), df_tfs[-1], freq='1T')
    res_mesh=cfg['KERNEL']['res_mesh']
    tfs=pd.date_range(
    df_tfs[0].strftime('%Y-%m-%d %H:%M'), df_tfs[-1].strftime('%Y-%m-%d %H:%M'), freq='1T')
    t1d=np.array([val.timestamp() for val in tfs])
    nt=len(tfs)
    nx,ny=lon_mesh.shape

    utils.write_log(f'{print_prefix}build uv mesh in {res_mesh}m resolution, size: ({nt} X {nx} X {ny})...')
    u,v=np.full((nt, nx, ny), np.nan),np.full((nt, nx, ny), np.nan)
    #u,v=2.0,0.8
    # Convert u to xarray DataArray and add dimensions and coordinates
    u = xr.DataArray(u, dims=('time', 'lat', 'lon'), 
                     coords={'time': tfs, 'lat': lat_mesh[:,0], 'lon': lon_mesh[0,:]})
    v=xr.DataArray(v, dims=('time', 'lat', 'lon'),
                     coords={'time': tfs, 'lat': lat_mesh[:,0], 'lon': lon_mesh[0,:]})
 
    nx,ny=lon_mesh.shape
    XLAT, XLONG, U10, V10, wrf_tfs=io.feed_uv(cfg, tfs)
    
    latsub, lonsub=find_area(XLAT, XLONG, lat_mesh, lon_mesh)
    
    utils.write_log(f'{print_prefix}locating and filling u and v into mesh...')
    for lat0, lon0 in zip(latsub, lonsub):
        ilat,ilon=locate_position(lat0, lon0, XLAT, XLONG)
        ilat_area, ilon_area=locate_position(lat0, lon0, lat_mesh, lon_mesh)
        for it, tf in enumerate(wrf_tfs):
            it_area=find_it(tf, t1d)
            u.values[it_area,ilat_area,ilon_area]=U10.values[it,ilat,ilon]
            v.values[it_area,ilat_area,ilon_area]=V10.values[it,ilat,ilon]
    
    if 'u' in df.columns.values:
        utils.write_log(f'{print_prefix} filling cruise wind observations into mesh...')
        uv_df = df.resample('1T').bfill()
        for idx, (tf, row) in enumerate(uv_df.iterrows()):
            ilat_area, ilon_area=locate_position(row['lat'], row['lon'], lat_mesh, lon_mesh)
            u.values[idx,ilat_area,ilon_area]=row['u']
            v.values[idx,ilat_area,ilon_area]=row['v']


    utils.write_log(f'{print_prefix} 3D interpolating...')
    u,v=interp_3d(u),interp_3d(v)
    return u,v
def interp_3d(da):
    da=da.interpolate_na(dim='lon', method="linear", fill_value="extrapolate") 
    da=da.interpolate_na(dim='lat', method="linear", fill_value="extrapolate") 
    da=da.interpolate_na(dim='time', method="linear", fill_value="extrapolate") 
    return da
def find_it(tf, t1d):
    tf=tf+datetime.timedelta(hours=8) # to localtime
    it= np.abs(t1d - tf.timestamp()).argmin()
    return it
def find_area(XLAT, XLONG, lat_mesh, lon_mesh):
    latmin, latmax=lat_mesh[:,0].min(), lat_mesh[:,0].max()
    lonmin, lonmax=lon_mesh[0,:].min(), lon_mesh[0,:].max()
    latminI, lonminI=locate_position(latmin, lonmin, XLAT, XLONG)
    latmaxI, lonmaxI=locate_position(latmax, lonmax, XLAT, XLONG)
    xlat_sub=XLAT.sel(south_north=slice(latminI.values, latmaxI.values), 
                    west_east=slice(lonminI.values, lonmaxI.values))
    
    xlon_sub=XLONG.sel(south_north=slice(latminI.values, latmaxI.values), 
                    west_east=slice(lonminI.values, lonmaxI.values))
    return xlat_sub.values.ravel(), xlon_sub.values.ravel()

def locate_position(lat0, lon0, lat_mesh, lon_mesh):
    # Calculate the index of the closest latitude and longitude points
    lat_index = np.abs(lat_mesh[:,0] - lat0).argmin()
    lon_index = np.abs(lon_mesh[0,:] - lon0).argmin()
    # Return the matrix index
    return lat_index, lon_index

def build_field(cfg, cruise_df, lat_mesh, lon_mesh):
    utils.write_log(f'{print_prefix}build concentration field...')
    conc, times=np.zeros(lat_mesh.shape), np.zeros(lat_mesh.shape)
    for idx, row in cruise_df.iterrows():
        lat_index, lon_index = locate_position(row['lat'], row['lon'], lat_mesh, lon_mesh)
        conc[lat_index, lon_index] += row[cfg['INPUT']['tgt_species']]
        times[lat_index, lon_index] +=1
    conc=conc/times
    return conc
    
    
def idw_interp_field(conc):   
    conc_interp=conc.copy()
    
    # Find indices of NaN values in conc matrix
    nan_idx = np.argwhere(np.isnan(conc))

    dis=5
    # Loop over NaN values and interpolate
    for i, j in nan_idx:
        # Find indices of non-NaN values within a radius of dis cells
        idx = np.argwhere(~np.isnan(conc[max(0, i-dis):i+dis+1, max(0, j-dis):j+dis+1]))

        # If there are no non-NaN values within the radius, skip this cell
        if len(idx) <=2:
            continue

        # Extract x, y, and z values for non-NaN values
        x = idx[:, 0] + max(0, i-dis)
        y = idx[:, 1] + max(0, j-dis)
        z = conc[x, y]
        # Interpolate to fill in NaN value
        conc_interp[i, j] = inverse_distance_interpolation(x, y, z, np.array([i]), np.array([j]))

    return conc_interp
# Define a function to perform inverse distance interpolation
def inverse_distance_interpolation(x, y, z, xi, yi, n=3):
    # Calculate distances between (x, y) and (xi, yi) points
    d = cdist(np.column_stack((x.ravel(), y.ravel())), np.column_stack((xi.ravel(), yi.ravel())))
    # Find n nearest points
    idx = np.argsort(d, axis=0)[:n, :]
    # Calculate inverse distance weights
    w = 1.0 / d[idx, :]

    # Normalize weights
    w /= np.sum(w, axis=0)

    # Calculate interpolated values
    zi = w.ravel().dot(z[idx].ravel())
    return zi

def paint2d(fn, var2d, lat_mesh, lon_mesh, minmax=(0, 500)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    pcm=ax.pcolormesh(lon_mesh, lat_mesh, var2d, cmap='jet',
                      vmin=minmax[0], vmax=minmax[1])
    # Add colorbar
    cbar = fig.colorbar(pcm)
    # Save the figure to a directory using the agg backend
    fig.savefig(fn, bbox_inches='tight', dpi=300, format='png')
    plt.close(fig)
