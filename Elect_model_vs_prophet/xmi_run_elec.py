# This file runs several configs that represent different system tests none of
# which should crash.

# Set up XMI on the system path. (alternatively install it).
import os
from xmi import xgate
from xmi import cmi 
import copy

tests_dir = os.path.dirname(os.path.abspath(__file__))


def get_test_path(name):
    return "{}/models/{}".format(tests_dir, name)

def rem_keys(d,kk):
    for s in kk:
        if s in d:
            del d[s]
    return

if True:
    # Run a standard multiplicative model (regression).
    cfgfile = get_test_path("stlf.cfg")
    par = cmi.parseModelCfg(cfgfile)
    par = cmi.add_default_par(par)

if True:
    # Ok start with something simple - just a neural network. 
    par1 = copy.deepcopy(par)
    rem_keys(par1,['node1','node2','node3','node5','node6','node7','node8','node9','node10'])
    logdir = par1['Paths']['log_directory'][0:-1] + '1/'
    par1['Paths']['log_directory']=logdir
    par1['node4']['input']=['0:y(location/calendar_hour/24);y(location/calendar_hour/25);y(location/calendar_hour/26);y(location/calendar_hour/27);y(location/calendar_hour/28);y(location/calendar_hour/29);y(location/calendar_hour/30);y(location/calendar_hour/31);y(location/calendar_hour/32);y(location/calendar_hour/33);y(location/calendar_hour/34);y(location/calendar_hour/35);y(location/calendar_hour/36);y(location/calendar_hour/37);y(location/calendar_hour/38);y(location/calendar_hour/39);y(location/calendar_hour/40);y(location/calendar_hour/41);y(location/calendar_hour/42);y(location/calendar_hour/43);y(location/calendar_hour/43);y(location/calendar_hour/44);y(location/calendar_hour/45);y(location/calendar_hour/46);y(location/calendar_hour/47);y(location/calendar_hour/48)']   
    par1['Model']['loss_fn'] = 'MSE'
    xgate.main(par1)
    
if True:
    # Ok start with something simple - just a neural network. 
    par1 = copy.deepcopy(par)
    rem_keys(par1,['node5','node6','node7','node8','node9','node10'])
    logdir = par1['Paths']['log_directory'][0:-1] + '2/'
    par1['Paths']['log_directory']=logdir
    par1['Model']['loss_fn'] = 'MSE'
    par1['node4']['act'] = 'sigmoid'
    xgate.main(par1)
