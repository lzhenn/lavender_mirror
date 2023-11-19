#/usr/bin/env python3
'''configuration funcs to get parameters from user'''

# ---imports---
import os
import configparser
from . import utils

# ---Module regime consts and variables---
print_prefix='lib.cfgparser>>'

# ---Classes and Functions---
def read_cfg(config_file):
    """ Simply read the config files """
    config=configparser.ConfigParser()
    config.read(config_file)
    return config

def write_cfg(cfg_hdl, config_fn):
    """ Simply write the config files """
    with open(config_fn, 'w') as configfile:
        cfg_hdl.write(configfile)

def cfg_get_varlist(cfg, key1, key2):
    varlist=cfg[key1][key2].split(',')
    varlist=[ele.strip() for ele in varlist]
    return varlist

# ---Unit test---
if __name__ == '__main__':
    pass