import numpy as np
import pickle
from pylab import *
from args_and_lambds import lambdas_dic
from nice_tcks import nice_ticks


model_type = 'VRY_4'
lambdas = lambdas_dic[model_type]

linew = 3
rc('font', size = 20) #fontsize of axis labels (numbers)
rc('axes', labelsize = 22, lw = linew) #fontsize of axis labels (symbols)
rc('lines', mew = 2, lw = linew, markeredgewidth = 2)
rc('patch', ec = 'k')
rc('xtick.major', pad = 7)
rc('ytick.major', pad = 7)
rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'
MS = 5      # marker size
MEW = 1     # marker edge width
lfs = 20            # legend font size

chi2_lat = pickle.load(open('chi2_wubp.p', "rb"))
file.close(open('chi2_wubp.p'))


### f rasters:
fname = model_type+'/'+'ftype1/fraster_'+model_type+'.p'
fdic1 = pickle.load(open(fname,'rb'))

fname = model_type+'/'+'ftype2/fraster_'+model_type+'.p'
fdic2 = pickle.load(open(fname,'rb'))

fname = model_type+'/'+'ftype5/fraster_'+model_type+'.p'
fdic3 = pickle.load(open(fname,'rb'))

phi0_raster = fdic1['phi0_raster']
f_raster1 = fdic1['f_raster']
f_raster2 = fdic2['f_raster']
f_raster3 = fdic3['f_raster']

figure(0)
plot(phi0_raster, f_raster1)
plot(phi0_raster, f_raster2)
plot(phi0_raster, f_raster3)

### chi2T2 rasters:
fname = model_type+'/'+'ftype1/TDTA_'+model_type+'.p'
TDTA_1 = pickle.load(open(fname,'rb'))

fname = model_type+'/'+'ftype2/TDTA_'+model_type+'.p'
TDTA_2 = pickle.load(open(fname,'rb'))

fname = model_type+'/'+'ftype5/TDTA_'+model_type+'.p'
TDTA_3 = pickle.load(open(fname,'rb'))


figure(1, figsize=(8, 6))
#errorbar(chi2_lat[:,0], chi2_lat[:,1], chi2_lat[:,2], ls='', marker='s', ms=MS, mew=MEW, mec='k', ecolor='grey', color='k', label = 'WuBp lattice')
plot(TDTA_1[0][:,0], lambdas[3]/(lambdas[2]*lambdas[0]**2.0)*TDTA_1[2], label='type 1', c='b')
plot(TDTA_2[0][:,0], lambdas[3]/(lambdas[2]*lambdas[0]**2.0)*TDTA_2[2], label='type 2', c='g', ls='--')
plot(TDTA_3[0][:,0], lambdas[3]/(lambdas[2]*lambdas[0]**2.0)*TDTA_3[2], label='type 3', c='r', ls=':')
errorbar(chi2_lat[:,0], chi2_lat[:,1], chi2_lat[:,2], ls='', marker='s', ms=MS, mew=MEW, mec='k', ecolor='grey', color='k', label = 'WuBp lattice')
axis([40, 160, 0, 0.14])
ax = subplot(111)
ax.set_yticks(np.arange(0, 0.14, 0.02))
legend(loc = 'best', numpoints=3, fontsize = lfs, frameon = 0, fancybox = 0, columnspacing = 1)
xlabel(r'$T \, [MeV\,]$')
ylabel(r'$\chi_2(T, \mu = 0)/T^{\, 2}$')
nice_ticks()
savefig(model_type+'/ftype1/pdfs/chis/chis_for_f_'+model_type+'.pdf')


show()