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


def rem_keys(d, kk):
    for s in kk:
        if s in d:
            del d[s]
    return


if True:
    # Run a standard multiplicative model (regression).
    cfgfile = get_test_path("stlf.cfg")
    par = cmi.parseModelCfg(cfgfile)
    par = cmi.add_default_par(par)

if False:
    # Simple 1 layer neural net with relu layer to more or less average the last 24 hours to get forecast
    par1 = copy.deepcopy(par)
    rem_keys(
        par1,
        [
            "node1",
            "node2",
            "node3",
            "node5",
            "node6",
            "node7",
            "node8",
            "node9",
            "node10",
            "node41",
            "node42",
        ],
    )
    logdir = par1["Paths"]["log_directory"][0:-1] + "1/"
    par1["Paths"]["log_directory"] = logdir
    par1["node4"]["input"] = [
        "0:y(location/calendar_hour/24);y(location/calendar_hour/25);y(location/calendar_hour/26);y(location/calendar_hour/27);y(location/calendar_hour/28);y(location/calendar_hour/29);y(location/calendar_hour/30);y(location/calendar_hour/31);y(location/calendar_hour/32);y(location/calendar_hour/33);y(location/calendar_hour/34);y(location/calendar_hour/35);y(location/calendar_hour/36);y(location/calendar_hour/37);y(location/calendar_hour/38);y(location/calendar_hour/39);y(location/calendar_hour/40);y(location/calendar_hour/41);y(location/calendar_hour/42);y(location/calendar_hour/43);y(location/calendar_hour/43);y(location/calendar_hour/44);y(location/calendar_hour/45);y(location/calendar_hour/46);y(location/calendar_hour/47);y(location/calendar_hour/48)"
    ]
    par1["node4"]["act"] = "Relu"
    par1["node4"]["breadth"] = "1"
    par1["Model"]["loss_fn"] = "MSE"
    par1["Model"]["trainrate"] = "1e-3"
    xgate.main(par1)

if False:
    # 2 layer Neural network only regressive inputs.
    par1 = copy.deepcopy(par)
    rem_keys(
        par1,
        [
            "node1",
            "node2",
            "node3",
            "node5",
            "node6",
            "node7",
            "node8",
            "node9",
            "node10",
        ],
    )
    logdir = par1["Paths"]["log_directory"][0:-1] + "2/"
    par1["Paths"]["log_directory"] = logdir
    par1["node4"]["input"] = [
        "0:y(location/calendar_hour/24);y(location/calendar_hour/25);y(location/calendar_hour/26);y(location/calendar_hour/27);y(location/calendar_hour/28);y(location/calendar_hour/29);y(location/calendar_hour/30);y(location/calendar_hour/31);y(location/calendar_hour/32);y(location/calendar_hour/33);y(location/calendar_hour/34);y(location/calendar_hour/35);y(location/calendar_hour/36);y(location/calendar_hour/37);y(location/calendar_hour/38);y(location/calendar_hour/39);y(location/calendar_hour/40);y(location/calendar_hour/41);y(location/calendar_hour/42);y(location/calendar_hour/43);y(location/calendar_hour/43);y(location/calendar_hour/44);y(location/calendar_hour/45);y(location/calendar_hour/46);y(location/calendar_hour/47);y(location/calendar_hour/48)"
    ]
    par1["Model"]["loss_fn"] = "MSE"
    par1["Model"]["trainrate"] = "1e-4"
    xgate.main(par1)

if False:
    par1 = copy.deepcopy(par)
    # 2 Lyaer neural network with regressors and embeddings of doy of week, location, and hour.
    rem_keys(par1, ["node5", "node6", "node7", "node8", "node9", "node10"])
    logdir = par1["Paths"]["log_directory"][0:-1] + "3/"
    par1["Paths"]["log_directory"] = logdir
    par1["Model"]["loss_fn"] = "MSE"
    par1["Model"]["trainrate"] = "1e-4"
    xgate.main(par1)


if False:
    # 2 layer net with dow,hour etc and modulated by a time of year kernel.
    par1 = copy.deepcopy(par)
    par1 = copy.deepcopy(par)
    rem_keys(par1, ["node7", "node8", "node9", "node10"])
    logdir = par1["Paths"]["log_directory"][0:-1] + "4/"
    par1["Paths"]["log_directory"] = logdir
    par1["Model"]["loss_fn"] = "MSE"
    par1["Model"]["trainrate"] = "1e-4"
    xgate.main(par1)

if False:
    # 2 layers with doy and covid kernel.
    par1 = copy.deepcopy(par)
    rem_keys(par1, ["node9", "node10"])
    logdir = par1["Paths"]["log_directory"][0:-1] + "5/"
    par1["Paths"]["log_directory"] = logdir
    par1["Model"]["loss_fn"] = "MSE"
    par1["Model"]["trainrate"] = "1e-4"
    xgate.main(par1)

if True:
    # all of the above with 2 outputs and gaussian likelihood.
    par1 = copy.deepcopy(par)
    logdir = par1["Paths"]["log_directory"][0:-1] + "6"
    par1["Paths"]["log_directory"] = logdir
    par1["node41"]["breadth"] = "2"
    xgate.main(par1)

# 1 - month ahead forecasts.
if False:
    # 2 layers with doy and covid kernel.
    par1 = copy.deepcopy(par)
    rem_keys(par1, ["node5", "node6", "node7", "node8", "node9", "node10"])
    logdir = par1["Paths"]["log_directory"][0:-1] + "7/"
    par1["node4"]["input"] = [
        "0:y(location/calendar_hour/720);y(location/calendar_hour/721);y(location/calendar_hour/722);y(location/calendar_hour/723);y(location/calendar_hour/724);y(location/calendar_hour/725);y(location/calendar_hour/726);y(location/calendar_hour/727);y(location/calendar_hour/728);y(location/calendar_hour/729);y(location/calendar_hour/730);y(location/calendar_hour/731);y(location/calendar_hour/732);y(location/calendar_hour/733);y(location/calendar_hour/734);y(location/calendar_hour/735);y(location/calendar_hour/736);y(location/calendar_hour/737);y(location/calendar_hour/738);y(location/calendar_hour/739);y(location/calendar_hour/740);y(location/calendar_hour/741);y(location/calendar_hour/742);y(location/calendar_hour/743);y(location/calendar_hour/744);y(location/calendar_hour/745)",
        "1",
        "2",
        "3",
    ]
    par1["Paths"]["log_directory"] = logdir
    par1["Model"]["loss_fn"] = "MSE"
    par1["Model"]["trainrate"] = "1e-4"
    xgate.main(par1)

if False:
    # 2 layers with doy and covid kernel.
    par1 = copy.deepcopy(par)
    rem_keys(par1, ["node9", "node10"])
    logdir = par1["Paths"]["log_directory"][0:-1] + "8/"
    par1["node4"]["input"] = [
        "0:y(location/calendar_hour/720);y(location/calendar_hour/721);y(location/calendar_hour/722);y(location/calendar_hour/723);y(location/calendar_hour/724);y(location/calendar_hour/725);y(location/calendar_hour/726);y(location/calendar_hour/727);y(location/calendar_hour/728);y(location/calendar_hour/729);y(location/calendar_hour/730);y(location/calendar_hour/731);y(location/calendar_hour/732);y(location/calendar_hour/733);y(location/calendar_hour/734);y(location/calendar_hour/735);y(location/calendar_hour/736);y(location/calendar_hour/737);y(location/calendar_hour/738);y(location/calendar_hour/739);y(location/calendar_hour/740);y(location/calendar_hour/741);y(location/calendar_hour/742);y(location/calendar_hour/743);y(location/calendar_hour/744);y(location/calendar_hour/745)",
        "1",
        "2",
        "3",
    ]
    par1["Paths"]["log_directory"] = logdir
    par1["Model"]["loss_fn"] = "MSE"
    par1["Model"]["trainrate"] = "1e-4"
    xgate.main(par1)

if False:
    # 2 layers with doy and covid kernel.
    par1 = copy.deepcopy(par)
    rem_keys(par1, ["node5", "node6", "node7", "node8", "node9", "node10"])
    logdir = par1["Paths"]["log_directory"][0:-1] + "9/"
    par1["node4"]["input"] = [
        "0:y(location/calendar_hour/2160);y(location/calendar_hour/2161);y(location/calendar_hour/2162);y(location/calendar_hour/2163);y(location/calendar_hour/2164);y(location/calendar_hour/2165);y(location/calendar_hour/2166);y(location/calendar_hour/2167);y(location/calendar_hour/2168);y(location/calendar_hour/2169);y(location/calendar_hour/2170);y(location/calendar_hour/2171);y(location/calendar_hour/2172);y(location/calendar_hour/2173);y(location/calendar_hour/2174);y(location/calendar_hour/2175);y(location/calendar_hour/2176);y(location/calendar_hour/2177);y(location/calendar_hour/2178);y(location/calendar_hour/2179);y(location/calendar_hour/2180);y(location/calendar_hour/2181);y(location/calendar_hour/2182);y(location/calendar_hour/2183);y(location/calendar_hour/2184);y(location/calendar_hour/2185)",
        "1",
        "2",
        "3",
    ]
    par1["Paths"]["log_directory"] = logdir
    par1["Model"]["loss_fn"] = "MSE"
    par1["Model"]["trainrate"] = "1e-4"
    xgate.main(par1)

if False:
    # 2 layers with doy and covid kernel.
    par1 = copy.deepcopy(par)
    rem_keys(par1, ["node9", "node10"])
    logdir = par1["Paths"]["log_directory"][0:-1] + "10/"
    par1["node4"]["input"] = [
        "0:y(location/calendar_hour/2160);y(location/calendar_hour/2161);y(location/calendar_hour/2162);y(location/calendar_hour/2163);y(location/calendar_hour/2164);y(location/calendar_hour/2165);y(location/calendar_hour/2166);y(location/calendar_hour/2167);y(location/calendar_hour/2168);y(location/calendar_hour/2169);y(location/calendar_hour/2170);y(location/calendar_hour/2171);y(location/calendar_hour/2172);y(location/calendar_hour/2173);y(location/calendar_hour/2174);y(location/calendar_hour/2175);y(location/calendar_hour/2176);y(location/calendar_hour/2177);y(location/calendar_hour/2178);y(location/calendar_hour/2179);y(location/calendar_hour/2180);y(location/calendar_hour/2181);y(location/calendar_hour/2182);y(location/calendar_hour/2183);y(location/calendar_hour/2184);y(location/calendar_hour/2185)",
        "1",
        "2",
        "3",
    ]
    par1["Paths"]["log_directory"] = logdir
    par1["Model"]["loss_fn"] = "MSE"
    par1["Model"]["trainrate"] = "1e-4"
    xgate.main(par1)
