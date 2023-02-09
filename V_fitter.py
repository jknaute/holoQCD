import numpy as np
from scipy.integrate import quad, odeint
from scipy.interpolate import splrep, splev
from amoba import amoeba
from args_and_lambds import args_dic, lambdas_dic
from Vtypes import Vs, dVs

def V(phi, *args):
    return Vs[args[0]](phi, *args[1])

def df_dphi(phi, *args):
    return dVs[args[0]](phi, *args[1])

# from Vtypes import Vs, dVs, dlogVs
# from H_expansions_G import get_horizon_expansions
# from args_and_lambds import args_dic, lambdas_dic

from rasterizer import rasterize
from time import time
import pickle

from backbone_g_5_mu0calc2_asmodule import chi2_mu0_calc_pt, TD_calc_mu0

from pylab import figure, plot, legend, show, axvline

def V_fit(p, data):
    metric_data = data[0]
    TD_data = data[1]
    phi0_raster = data[2]
    lat_data = data[3]
    lambdas = data[4]
    Vtype = data[5]
    fitwhat = data[5]

    V_args = [Vtype, list(p)]
    print V_args

    TDfunc_raster = np.zeros(len(phi0_raster))
    for i in range(0, len(phi0_raster)):
        TD_point = TD_data[i, :]
        metric_sol = metric_data[i]
        TDfunc_raster[i] =

    T_lat = lat_data[:, 0]
    chi2T2_lat = lat_data[:, 1]
    T_holo = TD_data[:, 0]
    #chi2T2_tck = splrep(T_holo[::-1], TDfunc_raster[::-1])
    chi2T2_tck = splrep(T_holo, TDfunc_raster)
    chi2T2_model = splev(T_lat, chi2T2_tck)
    print chi2T2_model
    chiq = np.sum((chi2T2_lat - chi2T2_model)**2.0)/np.float(len(phi0_raster))
    print 'chiq =', chiq
    chiq = np.log(chiq)
    print '- log chiq =', - chiq