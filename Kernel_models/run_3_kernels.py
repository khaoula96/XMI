# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
import h5py
import os
import sys

xmi_home = os.environ.get("XMI_HOME")
tests_dir = os.path.dirname(os.path.abspath(__file__))

if xmi_home not in sys.path:
    sys.path.insert(0, xmi_home)
import xmi
from xmi import xmi, cmi


"""
    3 kernels one of them is ones, 
    for kernel1 we feed x_inducing_points 
    and for kernel2 we feed the weights file

"""


def constractKernel(a, b):

    dist = gamma(a, b)  # working very well !!!
    x = np.linspace(1, 12, 100)  # 100 point for inducing kernels of dsi
    kernel = dist.pdf(x)
    kernel = kernel / kernel.max()
    return kernel


def adjustkernel(kernel):
    kernel[30:] = 0.38
    return kernel


def create_hdf5(inputs, filename):

    with h5py.File(filename, "w") as f:
        f.create_dataset("y_out", data=inputs["y"], dtype=np.float32)
        f.create_dataset("data_names", data=inputs["data_names"])
        f.create_dataset("data_shapes", data=inputs["data_shapes"])
        f.create_dataset("unhot_X_product_item", data=inputs["sku_id"])
        f.create_dataset("unhot_X_location_store", data=inputs["store_id"])
        f.create_dataset("unhot_X_day_since_introduction", data=inputs["dsi"])
        f.close()


def create_w_kernels(x, w, name, circular=True):

    if circular is True:
        a = 365
    if circular is False:
        a = 0
    else:
        a = circular
    count = 0
    for xi in range(0, w.shape[0]):
        dd = np.min(np.vstack((np.abs(x - xi), np.abs(np.abs(x - xi) - a))), axis=0)
        # for each data point assign a different width which is the distance to
        # its nearest neighbour ...
        sigman = []
        for xxi in x:
            temp = np.min(
                np.vstack((np.abs(x - xxi), np.abs(np.abs(x - xxi) - a))), axis=0
            )
            sigman.append((np.min(temp[temp > 0])) * 1)
        sigman = np.array(sigman)

        D = (
            1
            / np.sqrt(2 * np.pi * sigman**2)
            * np.exp((-((dd) ** 2)) / (2 * sigman**2))
        )
        # what if we just use the nearest 3 say ...
        w[count] = D / np.sum(D)
        count += 1
        np.savetxt(str(name), w, delimiter=",")


def create_syn_data(kernel1, kernel2, kernel3, size=100000):

    """y =  (a*sku_id + b store_id
    + c sku&store) * kernel1"""
    N = size
    sku_id = np.random.randint(0, 100, size=(N,))
    store_id = np.random.randint(0, 20, size=(N,))

    sku_store_weights = np.random.rand(100, 20) * 10
    sku_weights = (
        np.random.rand(
            100,
        )
        * 10
    )
    store_weights = (
        np.random.rand(
            20,
        )
        * 10
    )

    dsi = np.random.randint(0, 100, size=(N,))  # days since introduction.
    sku_kernel = np.zeros(N)
    sift1 = sku_id < 30
    sift2 = sku_id >= 30  # for 2 kernels 1 or 2
    sift3 = sku_id > 70

    sku_kernel[sift1] = 1
    sku_kernel[sift2] = 2
    sku_kernel[sift3] = 3

    y = np.zeros((N,))
    for i in range(N):
        y[i] = (
            sku_weights[sku_id[i]]
            + store_weights[store_id[i]]
            + sku_store_weights[sku_id[i]][store_id[i]]
        )

        if sku_kernel[i] == 1:
            addr = kernel1[dsi[i]]
        if sku_kernel[i] == 2:
            addr = kernel2[dsi[i]]
        if sku_kernel[i] == 3:
            addr = kernel3[dsi[i]]
        y[i] = y[i] * addr
    data = {}
    data["y"] = y.reshape((-1, 1))
    data["sku_id"] = sku_id.reshape((-1, 1))
    data["store_id"] = store_id.reshape((-1, 1))
    data["dsi"] = dsi.reshape((-1, 1))
    data["data_names"] = [
        "y_out",
        "unhot_X_product_item",
        "unhot_X_location_store",
        "unhot_X_day_since_introduction",
    ]
    data["data_shapes"] = [data["y"].shape[1], 100, 20, 100]

    return data


def run_data_creation(log_path):

    fyle = os.path.join(log_path, "data_kernels_123.hdf5")
    kernel1 = constractKernel(2, 1)
    kernel2 = constractKernel(4, 2)
    kernel3 = np.ones(kernel1.shape)

    syn_data = create_syn_data(kernel1, kernel2, kernel3)
    create_hdf5(syn_data, fyle)

    X1 = np.array([1, 5, 8, 15, 20, 28, 35, 60, 80, 99])

    X2 = np.array([1, 15, 20, 28, 35, 40, 55, 60, 80, 95])
    W1 = np.zeros((100, X1.shape[0]))

    # Feed a mix of both inducing points and weights to the gmi_modules
    k1_fyle = os.path.join(log_path, "kernel1_w.txt")
    create_w_kernels(X1, W1, k1_fyle, 0)
    x2_fyle = os.path.join(log_path, "x2.txt")
    np.savetxt(x2_fyle, X2, delimiter=",")

    return kernel1, kernel2


def get_test_path(name):
    return f"{xmi_home}/sys_tests/models/{name}"


def plot_predictions(par):
    hf = h5py.File(f"{xmi_home}/{par['Paths']['log_directory']}predictions.hdf5", "r")
    y_act = np.array(hf.get("y"))
    y_pred = np.array(hf.get("y_pred"))
    plt.plot(y_pred, y_act, "o", color="black")
    plt.show()


def getInducingPoints(par, name):

    kernel = pd.read_csv(
        f"{xmi_home}/{par['Paths']['log_directory']}coeffs/{name}_inducing_points.csv",
        header=None,
    )
    return kernel.values


def plot_kernels(oracle_ker, fama_ker, Plottype="Standard"):
    if Plottype == "norm":
        oracle_ker = oracle_ker / oracle_ker.max()
        fama_ker = fama_ker / fama_ker.max()
    plt.plot(oracle_ker, "r", label="Oracle kernel")
    plt.plot(fama_ker, "b", label="Fama kernel")
    plt.legend()
    plt.show()


def get_weights(par_file, name):
    weights = pd.read_csv(
        f"{xmi_home}/{par_file['Paths']['log_directory']}{name}",
        header=None,
    )
    return weights.values


if __name__ == "__main__":

    final_directory = os.path.join(tests_dir, r"kernel3")

    if not os.path.exists(final_directory):
        os.mkdir(final_directory)
    data_dir = final_directory

    oracle_kernel1, oracle_kernel2 = run_data_creation(data_dir)
    cfgfile_path = get_test_path("kernel3.cfg")

    par_file = cmi.parseModelCfg(cfgfile_path)
    xmi.main(cfgfile_path)
    plot_predictions(par_file)

    fama_induc_points1 = getInducingPoints(par_file, "kernel1")
    fama_induc_points2 = getInducingPoints(par_file, "kernel2")

    weights1 = get_weights(par_file, "kernel1_w.txt")
    weights2 = get_weights(par_file, "kernel2_w.txt")

    fama_kernel_1 = weights1.dot(fama_induc_points1)
    fama_kernel_2 = weights2.dot(fama_induc_points2)

    plot_kernels(oracle_kernel1, fama_kernel_1, "norm")
    plot_kernels(oracle_kernel2, fama_kernel_2, "norm")
