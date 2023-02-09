import numpy as np
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from scipy.interpolate import splrep, splev
from args_and_lambds import args_dic, lambdas_dic
from nice_tcks import nice_ticks

from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, LinearLocator

import pickle
from matplotlib.transforms import Bbox

linew = 3
rc('font', size = 20) #fontsize of axis labels (numbers)
rc('axes', labelsize = 20, lw = linew) #fontsize of axis labels (symbols)
rc('lines', mew = 2, lw = linew)
rc('patch', ec = 'k')

#rc('legend', fontsize = 16, frameon = True, fancybox = True, columnspacing = 1)
rc('legend', fontsize = 20, frameon = False, fancybox = False, columnspacing = 1)
#####################
model_type = 'G'
if model_type == 'G':
    fname = 'TD_gr_G_wmu0.p'
if model_type == 'no':
    fname = 'TD_gr_no.p'
TD_gr = pickle.load(open(fname, "rb"))
lambdas = lambdas_dic[model_type]

TD_grid = TD_gr[0]
phiPhi_grid = TD_gr[1]
TD_full = TD_gr[2]

phi0_pts = TD_full['phi0_pts']
Phi1_pts = TD_full['Phi1_pts']

fname = 'TD_gr_G_isen3.p'
TD_isen_dic = pickle.load(open(fname, "rb"))

isen_clrs = ['blue', 'grey', 'olive', 'purple', 'green']

print TD_isen_dic.keys()
epb_vals = np.sort(TD_isen_dic.keys())

print TD_isen_dic[10][0]

k = 0
for epb in epb_vals:
    TD_s = TD_scale_isen(TD_isen_dic[epb][0], lambdas)[0]
    print TD_s[:, 0]
    figure(1)
    plot(TD_s[:, 0], TD_s[:, 3]/TD_s[:, 0]**3.0, color = isen_clrs[k], label = r'$s/n = %2.1f$' %epb)

    figure(2)
    plot(TD_s[:, 2], TD_s[:, 3]/TD_s[:, 0]**3.0, color = isen_clrs[k], label = r'$s/n = %2.1f$' %epb)

    k += 1

zoom = 1
figure(1)
xlabel(r'$T \, [MeV]$')
ylabel(r'$n/T^{\,3}$')

if not zoom:
    axis([0, 1000, 0, 5])
    ax = subplot(111)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    legend(loc = 'best', fontsize = 18)
    nice_ticks()
    ax = subplot(111)
    ax.set_position(Bbox([[0.10, 0.14], [0.95, 0.95]]))
else:
    axis([0, 1000, 0, 0.5])
    ax = subplot(111)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    #legend(loc = 'best', fontsize = 18, ncol = 2)
    nice_ticks()
    ax = subplot(111)
    ax.set_position(Bbox([[0.12, 0.14], [0.95, 0.95]]))
#savefig('G_isentropic_eos_nT3_T_zoom'+str(zoom)+'.pdf')

figure(2)
xlabel(r'$\mu [MeV]$')
ylabel(r'$n/T^{\,3}$')

if not zoom:
    axis([0, 2500, 0, 5])
    ax = subplot(111)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_minor_locator(MultipleLocator(125))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    legend(loc = 'best', fontsize = 18)
    nice_ticks()
    ax = subplot(111)
    ax.set_position(Bbox([[0.10, 0.14], [0.95, 0.95]]))
else:
    axis([0, 1000, 0, 0.5])
    ax = subplot(111)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    #legend(loc = 'best', fontsize = 18, ncol = 2)
    nice_ticks()
    ax = subplot(111)
    ax.set_position(Bbox([[0.12, 0.14], [0.95, 0.95]]))
#savefig('G_isentropic_eos_nT3_mu_zoom'+str(zoom)+'.pdf')

show()