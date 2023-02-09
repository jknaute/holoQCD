import numpy as np
from scipy.interpolate import splrep, splev

from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp, subplots, gcf, tight_layout, subplots_adjust, grid, title
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, LinearLocator
from matplotlib.transforms import Bbox
from nice_tcks import nice_ticks
from scipy.misc import derivative as deriv

import pickle

linew = 3
rc('font', size = 20) #fontsize of axis labels (numbers)
rc('axes', labelsize = 22, lw = linew) #fontsize of axis labels (symbols)
rc('lines', mew = 2, lw = linew, markeredgewidth = 2)
rc('patch', ec = 'k')
rc('xtick.major', pad = 7)
rc('ytick.major', pad = 7)
rc('legend', frameon = False)

eos_lat = pickle.load(open('QG_latticedata_WuB.p', "rb"))
file.close(open('QG_latticedata_WuB.p'))
print eos_lat['T']
mu0_pT4_tck = splrep(eos_lat['T'], eos_lat['pT4'])
mu0_sT3_tck = splrep(eos_lat['T'], eos_lat['sT3'])

chi2_lat = pickle.load(open('chi2_wubp.p', "rb"))
file.close(open('chi2_wubp.p'))

T = chi2_lat[:, 0]
chi2hat = chi2_lat[:, 1]
mu0_chi2hat_tck = splrep(T, chi2hat)

chi4_lat = pickle.load(open('chi4_wubp.p', "rb"))
file.close(open('chi4_wubp.p'))

Tchi4 = chi4_lat[:,0]
chi4 = chi4_lat[:, 1]
mu0_chi4_tck = splrep(Tchi4, chi4)

def pT4_mu0(T, pT4_tck):
    return splev(T, pT4_tck)

def pT4_nlo(T, mu, chi2hat_tck):
    return 1.0/2.0*mu**2.0/T**2.0*splev(T, chi2hat_tck)

def pT4_nnlo(T, mu, chi4_tck):
    return 1.0/24.0*mu**4.0/T**4.0*splev(T, chi4_tck)

def sT3_mu0(T, sT3_tck):
    return splev(T, sT3_tck)

def sT3_nlo(T, mu, chi2hat_tck):
    dchi2hat_dT = splev(T, chi2hat_tck, der = 1)
    dchi2_dT = 2.0*T*splev(T, chi2hat_tck) + T**2.0*dchi2hat_dT
    s_nlo = 1.0/2.0*mu**2.0*dchi2_dT
    sT3_nlo = s_nlo/T**3.0
    return sT3_nlo

def sT3_nnlo(T, mu, chi4_tck):
    s_nnlo = 1.0/24.0*mu**4.0*splev(T, chi4_tck, der = 1)
    sT3_nnlo = s_nnlo/T**3.0
    return sT3_nnlo

def nT3_mu0(T):
    return 0

def nT3_nlo(T, mu, chi2hat_tck):
    return mu/T*splev(T, chi2hat_tck)

def nT3_nnlo(T, mu, chi4_tck):
    return 1.0/6.0*mu**3.0/T**3.0*splev(T, chi4_tck)

def pT4_lat(T, mu, pT4_tck, chi2hat_tck, chi4_tck, order):
    if order == 0:
        return pT4_mu0(T, pT4_tck)
    elif order == 1:
        return pT4_mu0(T, pT4_tck) + pT4_nlo(T, mu, chi2hat_tck)
    elif order == 2:
        return pT4_mu0(T, pT4_tck) + pT4_nlo(T, mu, chi2hat_tck) + pT4_nnlo(T, mu, chi4_tck)

def sT3_lat(T, mu, sT3_tck, chi2hat_tck, chi4_tck, order):
    if order == 0:
        return sT3_mu0(T, sT3_tck)
    elif order == 1:
        return sT3_mu0(T, sT3_tck) + sT3_nlo(T, mu, chi2hat_tck)
    elif order == 2:
        return sT3_mu0(T, sT3_tck) + sT3_nlo(T, mu, chi2hat_tck) + sT3_nnlo(T, mu, chi4_tck)

def nT3_lat(T, mu, chi2hat_tck, chi4_tck, order):
    if order == 0:
        return 0
    elif order == 1:
        return nT3_nlo(T, mu, chi2hat_tck)
    elif order == 2:
        return nT3_nlo(T, mu, chi2hat_tck) + nT3_nnlo(T, mu, chi4_tck)

def get_contour_verts(cn):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
                paths.append(np.vstack(xy))
        contours.append(paths)

    return contours

T_raster = np.linspace(140.0, 300.0, 200)
mu_raster = np.linspace(0.0, 500.0, 400)

sn_grid = np.zeros((len(T_raster), len(mu_raster)))
for i in range(0, len(T_raster)):
    for j in range(0, len(mu_raster)):
        T = T_raster[i]
        mu = mu_raster[j]#/1.5
        sn_grid[i, j] = sT3_lat(T, mu, mu0_sT3_tck, mu0_chi2hat_tck, mu0_chi4_tck, 2)/nT3_lat(T, mu, mu0_chi2hat_tck,
                                                                                         mu0_chi4_tck, 2)

sn_levels = [420.0, 144.0, 94.0, 68.0, 48.5][::-1]
#sn_levels = [94.0, 68.0, 48.5][::-1]
figure(1)
sn_cnts_QCS = contour(mu_raster, T_raster, sn_grid, levels = sn_levels, cmap = cm.autumn)
#sn_cnts_QCS = contour(mu_raster, T_raster, np.transpose(sn_grid), levels = sn_levels, cmap = cm.autumn)
nice_ticks()
ax = subplot()
#ax.set_position(Bbox([[0.16, 0.16], [0.8, 0.95]]))
#ax.set_position(Bbox([[0.2, 0.24], [0.7, 0.9]]))
cb = colorbar(ax = ax)
cb.ax.get_children()[4].set_linewidths(8)
subplots_adjust(left = 0.16, bottom = 0.16)

axis([0, 500, 50, 300])
xlabel(r'$\mu \, [MeV]$')
ylabel(r'$T \, [MeV]$')
title(r'$s/n = const$ up to nnlo', y = 1.02)

T_raster = np.linspace(130, 500, 500)
mu_levels = np.arange(0, 1600.6, 400)
for mu in mu_levels:
    figure(2)
    pT4_raster = pT4_lat(T_raster, mu, mu0_pT4_tck, mu0_chi2hat_tck, mu0_chi4_tck, 1)
    plot(T_raster, pT4_raster)
    figure(3)
    sT3_raster = sT3_lat(T_raster, mu, mu0_sT3_tck, mu0_chi2hat_tck, mu0_chi4_tck, 1)
    plot(T_raster, sT3_raster)
    figure(4)
    nT3_raster = nT3_lat(T_raster, mu, mu0_chi2hat_tck, mu0_chi4_tck, 1)
    plot(T_raster, nT3_raster)

show()
