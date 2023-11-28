#!/usr/bin/env python3
from . import  utils, mesh 
import numpy as np
import datetime
print_prefix='lib.kernel>>'
# CONSTANT
R_EARTH=6371000
DIS2LAT=180/(np.pi*R_EARTH)        #Distance to Latitude
CONST={'a':R_EARTH,'dis2lat':DIS2LAT}

def lagrun(cfg, epi_df, u, v, lat2d, lon2d):
    '''
    lagrun
    '''
    ptcls=Particles(cfg, epi_df, lat2d, lon2d)
    ptcls.march(u,v)

    conc, times=np.zeros(lat2d.shape), np.zeros(lat2d.shape)
    latmin,latmax= lat2d.min(), lat2d.max()
    lonmin,lonmax= lon2d.min(), lon2d.max()
    utils.write_log(f'{print_prefix}Aggregating conc...')
    for idx in range(ptcls.nptcls):
        conc0=ptcls.conc[idx]
        for idt in range(ptcls.nsteps):
            if ptcls.xlats[idx,idt]<latmin or ptcls.xlats[idx,idt]>latmax:
                continue
            if ptcls.xlons[idx,idt]<lonmin or ptcls.xlons[idx,idt]>lonmax:
                continue
            lat_index, lon_index = mesh.locate_position(
                ptcls.xlats[idx,idt], ptcls.xlons[idx,idt], lat2d, lon2d)
            conc[lat_index, lon_index] += conc0
            times[lat_index, lon_index] +=1
    conc=conc/times
    return conc, times 


class Particles:

    '''
    Construct air parcel array

    Attributes

    part_id:        int, partical id
    xlon, xlat:     float, longitude/latitude of the particles
    conc:           float, mass of tracer 

    Methods
    -----------


    '''
    
    def __init__(self, cfg, df, lat2d, lon2d):
        """ construct air parcel obj """
        self.nptcls=df.shape[0]
        self.nsteps=int(cfg['KERNEL']['nsteps'])
        self.dt=int(cfg['KERNEL']['dt'])
        self.part_id=np.arange(self.nptcls,dtype=np.int32)
        self.conc=np.zeros(self.nptcls)
        self.xlons=np.zeros((self.nptcls,self.nsteps),dtype=np.float32)
        self.xlats=self.xlons.copy()
        self.xtimes=[]
        for idx, (tf, row) in enumerate(df.iterrows()):
            self.xlons[idx,0],self.xlats[idx,0]=row['lon'],row['lat']
            self.xtimes.append(tf)
            self.conc[idx]=row[cfg['INPUT']['tgt_species']]
    
        utils.write_log(
            print_prefix+'array with %d parcels initiated!' % self.nptcls)

    def march(self,u,v):
        """ march particles """
        dt=self.dt 
        u4d,v4d=u.values,v.values
        lat1d,lon1d=u.lat.values,u.lon.values
        t1d=np.array([val.tolist()/1e9 for val in u.time.values])
        utils.write_log(
                print_prefix+'Marching %d parcels for %d seconds' % (
                self.nptcls, dt*self.nsteps))
            

        for idx in range(self.nptcls):
            tf=self.xtimes[idx]
            it=tf.timestamp()
            for idt in range(self.nsteps-1):
                ilat,ilon=self.xlats[idx,idt],self.xlons[idx,idt]
                idlat=get_closest_idx_lsearch(lat1d, ilat)
                idlon=get_closest_idx_lsearch(lon1d, ilon)
                idtime=get_closest_idx_lsearch(t1d, it)
                u0=u4d[idtime,idlat,idlon]
                v0=v4d[idtime,idlat,idlon]
                dx=u0*dt
                dlon=dx*180/(CONST['a']*np.sin(np.pi/2-np.radians(ilat))*np.pi)
                dy=v0*dt
                dlat=dy*CONST['dis2lat']
                
                # update
                self.xlats[idx,idt+1]=self.xlats[idx,idt]+dlat
                self.xlons[idx,idt+1]=self.xlons[idx,idt]+dlon
                it = it+dt

def get_closest_idx_lsearch(l1d, tgt_value):
    """
        Find the nearest idx in l1d (linear search)
    """
    return np.abs(l1d - tgt_value).argmin()