#/usr/bin/env python3
"""
    Commonly used utilities

    Function    
    ---------------
    throw_error(msg):
        throw error and exit
    
    write_log(msg, lvl=20):
        write logging log to log file
    
    parse_tswildcard(tgt_time, wildcard):
        parse string with timestamp wildcard 
        to datetime object

"""
# ---imports---
import logging
import numpy as np


# ---Module regime consts and variables---
print_prefix='lib.utils>>'

DEG2RAD=np.pi/180.0
WD_DIC={'N':  0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,
            'E': 90.0, 'ESE':112.5, 'SE':135.0, 'SSE':157.5,
            'S':180.0, 'SSW':202.5, 'SW':225.0, 'WSW':247.5,
            'W':270.0, 'WNW':292.5, 'NW':315.0, 'NNW':337.5}
    

def wswd2uv(ws, wd):
    """ convert wind component to UV """
    # below test valid wind direction input
    wd_error=False
    if np.isnan(wd) or np.isnan(ws):
        return (np.nan, np.nan)
    try: 
        wd_int=int(wd)
        if wd_int>=0.0 and wd_int<=360.0:
            wd_rad=wd_int*DEG2RAD
        else:
            wd_error=True
    except ValueError:
        try:
            wd_rad=WD_DIC[wd]*DEG2RAD
        except KeyError:
            wd_error=True
    
    if wd_error:
        throw_error('utils.wswd2uv>>Invalid wind direction input! '
                +f'with record: wind_dir={wd}')

    u=-np.sin(wd_rad)*ws
    v=-np.cos(wd_rad)*ws
    
    return (u,v)

# ---Classes and Functions---
def throw_error(msg):
    '''
    throw error and exit
    '''
    logging.error(msg)
    exit()

def write_log(msg, lvl=20):
    '''
    write logging log to log file
    level code:
        CRITICAL    50
        ERROR   40
        WARNING 30
        INFO    20
        DEBUG   10
        NOTSET  0
    '''

    logging.log(lvl, msg)

def parse_tswildcard(tgt_time, wildcard):
    '''
    parse string with timestamp wildcard to datetime object
    '''
    seg_str=wildcard.split('@')
    parsed_str=''
    for seg in seg_str:
        if seg.startswith('%'):
            parsed_str+=tgt_time.strftime(seg)
        else:
            parsed_str+=seg
    return parsed_str


# ---Unit test---
if __name__ == '__main__':
    pass

