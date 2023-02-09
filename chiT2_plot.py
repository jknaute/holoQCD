import numpy as np

from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp, subplots, gcf, tight_layout, subplots_adjust, grid
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, LinearLocator
from matplotlib.transforms import Bbox

import pickle

from args_and_lambds import args_dic, lambdas_dic
from rasterizer import rasterize

linew = 3
rc('font', size = 20) #fontsize of axis labels (numbers)
rc('axes', labelsize = 22, lw = linew) #fontsize of axis labels (symbols)
rc('lines', mew = 2, lw = linew, markeredgewidth = 2)
rc('patch', ec = 'k')
rc('xtick.major', pad = 7)
rc('ytick.major', pad = 7)

model_type = 'G'

TDTA_G = pickle.load(open('TDTA_G.p', "rb"))
file.close(open('TDTA_G.p'))

#TDTA_no = pickle.load(open('TDTA_no.p', "rb"))
#file.close(open('TDTA_no.p'))

##suffx = 'Vno_ftanh'
#suffx = 'VG_ftanh_fit'
#fname = 'TDTA_'+suffx+'.p'
#TDTA_new = pickle.load(open(fname, "rb"))
##print TDTA_new.keys()
#file.close(open(fname))
#phi0_raster_n = TDTA_new[1]

chi2_lat = pickle.load(open('chi2_wubp.p', "rb"))
file.close(open('chi2_wubp.p'))

chiT2_G = TDTA_G[2]
T_raster_G = TDTA_G[0][:, 0]
#chiT2_no = TDTA_no[2]
#T_raster_no = TDTA_no[0][:, 0]
lambdas_G = lambdas_dic['G']
#lambdas_no = lambdas_dic['no']

chiT2lamb_G = lambdas_G[3]/(lambdas_G[2])/lambdas_G[0]**2.0
#chiT2lamb_no = lambdas_no[3]/(lambdas_no[2])/lambdas_no[0]**2.0

chi2scale = chiT2lamb_G
#print TDTA_new[2]*chi2scale

figure(1)
plot(T_raster_G, chiT2_G*chiT2lamb_G, color = 'black', label = 'Gubser')
#plot(T_raster_no, chiT2_no*chiT2lamb_no, color = 'blue', label = 'Noronha')
#plot(TDTA_new[0][:,0], TDTA_new[2]*chi2scale, color = 'green', label = r'$V_{G}, f_{tanh}$')
scatter(chi2_lat[:, 0], chi2_lat[:, 1], marker = 'x', color = 'k', s = 60, lw = 2, label = 'WuBp data')
axis([100, 450, 0, 0.4])
legend(loc = 'lower right', frameon = False)

figure(2)
plot(T_raster_G, chiT2_G*chiT2lamb_G, color = 'black', label = 'Gubser')
#plot(T_raster_no, chiT2_no*chiT2lamb_no, color = 'blue', label = 'Noronha')
#plot(TDTA_new[0][:,0], TDTA_new[2]*chi2scale, color = 'green', label = r'$V_{G}, f_{tanh}$')
scatter(chi2_lat[:, 0], chi2_lat[:, 1], marker = 'x', color = 'k', s = 60, lw = 2, label = 'WuBp data')
axis([50, 2000, 0, 0.4])
legend(loc = 'lower right', frameon = False)

#figure(fig)
MLs = {1:[50, 10, 0.05, 0.01], 2:[500, 100, 0.05, 0.01]}
savenames = {1:'chiT2_',2:'chiT2_large_'}

for fig in [1,2]:
    figure(fig)
    xlabel(r'$T \, [MeV\,]$')
    ylabel(r'$\chi_2(T, \mu = 0)/T^{\,2}$')
    ax = subplot(111)
    ax.xaxis.set_major_locator(MultipleLocator(MLs[fig][0]))
    ax.xaxis.set_minor_locator(MultipleLocator(MLs[fig][1]))

    ax.yaxis.set_major_locator(MultipleLocator(MLs[fig][2]))
    ax.yaxis.set_minor_locator(MultipleLocator(MLs[fig][3]))
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(10)
        l.set_markeredgewidth(2.0)
    for l in ax.yaxis.get_minorticklines() + ax.xaxis.get_minorticklines():
        l.set_markersize(5)
        l.set_markeredgewidth(1.5)

    ax.set_position(Bbox([[0.16, 0.14], [0.95, 0.95]]))
    savefig(savenames[fig]+model_type+'.pdf')
    #legend(loc = 'best', fontsize = 18, frameon = False)
    #legend(loc = 'best', frameon = False)
#grid(which = 'both')
#'savefig('chiT2'+savenames[fig]+'.pdf')

#chiT2_integral_raster = TDTA_new[3]
#f_raster = TDTA_new[4]
#figure(10)
#plot(phi0_raster_n, TDTA_new[0][:,1]/TDTA_new[0][:,0]**3.0, label = r'$s/T^{\,3}$')
#plot(phi0_raster_n, TDTA_new[2]*chi2scale, label = r'$\chi_2/T^{\,2}$')
#plot(phi0_raster_n, 0.1*1.0/chiT2_integral_raster, label = r'$0.1/int$')
#plot(phi0_raster_n, f_raster, label = r'$f(\phi)$')
#plot(phi0_raster_n, 1.0/f_raster, label = r'$1/f(\phi)$')
#plot(phi0_raster_n, 0.1*TDTA_new[0][:,0], label = r'$0.1T [MeV]$')
#xlabel(r'$\phi_0$')
#legend(frameon = False, fontsize = 16)
#ax = subplot(111)
#ax.set_ylim(0, 25)
show()