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
    conc, times, weights=np.zeros(lat2d.shape), np.zeros(lat2d.shape), np.zeros(lat2d.shape)
    latmin,latmax= lat2d.min(), lat2d.max()
    lonmin,lonmax= lon2d.min(), lon2d.max()

    nens=1
    if ptcls.turbflag:
        nens=int(cfg['KERNEL']['turb_nens'])
    diff_weight=diff_mode(ptcls.diff_mode, ptcls.dt, ptcls.nsteps)
    for ntime in range(nens):
        ptcls.march(u,v)
        utils.write_log(f'{print_prefix}Aggregating conc, round {ntime+1}/{nens}...')
        for idx in range(ptcls.nptcls):
            conc0=ptcls.conc[idx]
            for idt in range(ptcls.nsteps):
                if ptcls.xlats[idx,idt]<latmin or ptcls.xlats[idx,idt]>latmax:
                    continue
                if ptcls.xlons[idx,idt]<lonmin or ptcls.xlons[idx,idt]>lonmax:
                    continue
                lat_index, lon_index = mesh.locate_position(
                    ptcls.xlats[idx,idt], ptcls.xlons[idx,idt], lat2d, lon2d)
                conc[lat_index, lon_index] += conc0*diff_weight[idt]
                weights[lat_index, lon_index] += diff_weight[idt]
                times[lat_index, lon_index] +=1
    weights=weights/times/nens    
    conc=conc/times
    return conc, times, weights 

def diff_mode(diff_mode, dt, nsteps, half=1200):
    dt=abs(dt)
    tstep=np.linspace(0, nsteps*dt, nsteps)
    if diff_mode=='conservative':
        diff_weight=np.ones(nsteps)
    elif diff_mode=='linear':
        diff_weight=np.zeros(nsteps)
        n0_step=int(half*2/dt)
        if n0_step<nsteps:
            diff_weight[:n0_step]=np.linspace(1,0,n0_step)
        else:
            diff_weight=np.linspace(1,0,nsteps)
    elif diff_mode=='exponential':
        diff_weight=np.exp2(-tstep/half)
    elif diff_mode=='gaussian':
        k=-1.0*np.exp(0.5)/(half**2)
        diff_weight=np.exp(k*tstep**2)
    else:
        utils.throw_error(f'unknown diff mode:{diff_mode}')
    return diff_weight

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
        self.turbflag=cfg['KERNEL'].getboolean('turb_flag')
        if self.turbflag:
            self.turblv=int(cfg['KERNEL']['turb_lv'])
            if self.turblv>5:
                self.turblv=5
            elif self.turblv<1:
                self.turblv=1
        self.diff_mode=cfg['KERNEL']['diff_mode']
        self.part_id=np.arange(self.nptcls,dtype=np.int32)
        self.conc=np.zeros(self.nptcls)
        self.dxs=np.zeros((self.nptcls,self.nsteps),dtype=np.float32)
        self.dys=np.zeros((self.nptcls,self.nsteps),dtype=np.float32)
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
        dlatr=180/(CONST['a']*np.sin(np.pi/2-np.radians(lat1d.mean()))*np.pi)
        utils.write_log(
                print_prefix+'Marching %d parcels for %d seconds' % (
                self.nptcls, dt*self.nsteps))
        rand_ufactor=rand_vfactor=np.zeros((self.nptcls,self.nsteps))
        if self.turbflag:
            rand_ufactor=np.random.normal(size=(self.nptcls,self.nsteps))*0.2*self.turblv
            rand_vfactor=rand_ufactor[::-1,::-1]
        for idx in range(self.nptcls):
            tf=self.xtimes[idx]
            it=tf.timestamp()
            for idt in range(self.nsteps-1):
                ilat,ilon=self.xlats[idx,idt],self.xlons[idx,idt]
                idlat=get_closest_idx_lsearch(lat1d, ilat)
                idlon=get_closest_idx_lsearch(lon1d, ilon)
                idtime=get_closest_idx_lsearch(t1d, it)
                u0=u4d[idtime,idlat,idlon]*(1+rand_ufactor[idx,idt])
                v0=v4d[idtime,idlat,idlon]*(1+rand_vfactor[idx,idt])
                
                dx=u0*dt
                dlon=dx*dlatr
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