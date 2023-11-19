#!/usr/bin/env python3
'''
Date: May 27, 2023
lavender_mirror is

This is the main script to drive the model

History:
May 27, 2023 --- Kick off the project

L_Zealot
'''
import sys, os
import logging, logging.config
from .lib import cfgparser, utils, io, mesh, kernel


package_name = 'lavender_mirror'
print_prefix='run_model>>'
# path to the top-level handler
CWD=sys.path[0]

# path to this module
#MWD=os.path.split(os.path.realpath(__file__))[0]

def waterfall():
    '''
    Waterfall rundown!
    '''

    _setup_logging()

    cfg=cfgparser.read_cfg('config.case.ini')

    utils.write_log(f'{print_prefix}Main pipeline...')
    # read data, setup mesh
    episode_df=io.read_episode(cfg)
    lon2d, lat2d=mesh.build_meshgrid(cfg)
    u,v=mesh.build_uvmesh(cfg, episode_df.index, lat2d, lon2d)
    conc_dry=mesh.build_field(episode_df, lat2d, lon2d)
    
    # traced conc distribution
    conc_trace, times=kernel.lagrun(cfg, episode_df, u, v, lat2d, lon2d)
    conc_trace=mesh.idw_interp_field(conc_trace)
    
    # output
    io.outnc(
        cfg, lat2d, lon2d, 
        {'conc_dry':conc_dry, 'conc_trace':conc_trace, 'tracer_counts':times})
    
    # paint
    fig_path=cfg['OUTPUT']['fig_dir']
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    mesh.paint2d(os.path.join(fig_path,'conc_dry.png'), conc_dry, lat2d, lon2d)
    mesh.paint2d(os.path.join(fig_path,'conc_trace.png'), conc_trace, lat2d, lon2d)
    mesh.paint2d(os.path.join(fig_path,'conc_trace_times.png'), times, lat2d, lon2d,minmax=(0, 50))

def _setup_logging():
    """
    Configures the logging module using the
    'config.logging.ini' file in the installed package.
    """
    resource_path = 'lavender_mirror/conf/config.logging.ini'
    logging.config.fileConfig(
        resource_path, disable_existing_loggers=False)
if __name__ == '__main__':
    waterfall()