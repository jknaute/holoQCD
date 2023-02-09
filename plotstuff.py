import numpy as np
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from scipy.interpolate import splrep, splev
from matplotlib.transforms import Bbox

from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp, subplots, gcf, tight_layout, subplots_adjust

from Vtypes import Vs, dVs
from ftypes import fs, dfs
from rasterizer import rasterize

phi0_raster = np.linspace(0, 20.0, 600)

# nrm = np.cosh(12.0/5.0)
# scl = 6.0/5.0
# shft = 2.0
# f_args = ['f_I', np.array([nrm, scl, shft])]

gamma = 0.606
b = 0.703
c_4 = -0.1
c_6 = 0.0034
V_args = ['V_VI',np.array([gamma, b, c_4, c_6])]

nrm = 1.0/3.0*np.cosh(0.69)
scl = 1.2
shft = 0.69/1.2
n2 = 2.0/3.0
efac = -100.0
f_args = ['f_no', np.array([nrm, scl, shft, n2, efac])]

def V(phi, *args):
    return Vs[args[0]](phi, *args[1])

def dV_dphi(phi, *args):
    return dVs[args[0]](phi, *args[1])

def f(phi, *args):
    return fs[args[0]](phi, *args[1])

def df_dphi(phi, *args):
    return dfs[args[0]](phi, *args[1])

figure(1)
plot(phi0_raster, rasterize(f, phi0_raster, *f_args)[1])

figure(2)
plot(phi0_raster, rasterize(dV_dphi, phi0_raster, *V_args)[1]/rasterize(V, phi0_raster, *V_args)[1])
show()