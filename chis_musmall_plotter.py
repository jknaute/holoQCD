import numpy as np
import pickle

from pylab import figure, plot, legend, show, semilogy, loglog, axis, subplot, xlabel, ylabel, rc, savefig, errorbar
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, LinearLocator

from nice_tcks import nice_ticks
from args_and_lambds import args_dic, lambdas_dic


linew = 3
rc('font', size = 20) #fontsize of axis labels (numbers)
rc('axes', labelsize = 22, lw = linew) #fontsize of axis labels (symbols)
rc('lines', mew = 2, lw = linew, markeredgewidth = 2)
rc('patch', ec = 'k')
rc('xtick.major', pad = 7)
rc('ytick.major', pad = 7)
rc('legend', frameon = False)

MS = 5      # marker size
MEW = 1     # marker edge width




model_type = 'VRY_4'
ftype = args_dic['ftype'][model_type]


fname = model_type+'/'+ftype+'/TDTA_'+model_type+'.p'
TDTA = pickle.load(open(fname, "rb"))
file.close(open(fname))

V_args = args_dic['V'][model_type]
f_args = args_dic['f'][model_type]
lambdas = lambdas_dic[model_type]

TD = TDTA[0]
phi0_raster = TDTA[1]
T_raster = TD[:, 0]
chiT2_raster_fm = TDTA[2]
#print phi0_raster, T_raster

fname = model_type+'/'+ftype+'/chis_list_'+model_type+'2.p'
chis_dic = pickle.load(open(fname, "rb"))
file.close(open(fname))
Phi_1rs = np.sort(chis_dic.keys())#[:1]
print Phi_1rs


#+++++++++++++++++++++++ lattice data:
chi2_lat = pickle.load(open('chi2_wubp.p', "rb"))
file.close(open('chi2_wubp.p'))
chi4_lat = pickle.load(open('chi4_wubp.p', "rb"))
file.close(open('chi4_wubp.p'))
#+++++++++++++++++++++++


for key in chis_dic.keys():
    print chis_dic[key][1]

for i in range(0, len(Phi_1rs[:1])):
    #print chis_dic[Phi_1rs[i]]
    print Phi_1rs[i]
    chi_2_raster = chis_dic[Phi_1rs[i]][0]
    chi_3_raster = chis_dic[Phi_1rs[i]][1]
    chi_4_raster = chis_dic[Phi_1rs[i]][2]
    muval = chis_dic[Phi_1rs[i]][3]
    print i, chi_3_raster



savenames = {1:'chi_2_', 2:'chi_3_', 3:'chi_4_'}

### chi_2
figure(1)
errorbar(chi2_lat[:,0], chi2_lat[:,1],chi2_lat[:,2], ls='', marker='s', ms=MS, mew=MEW, mec='k', c='k', label = 'WuBp data')
plot(T_raster, chi_2_raster/T_raster**2.0*lambdas[3]/lambdas[2], label = r'$\mu = %2.4f \, [MeV]$ numerical' %muval)
plot(T_raster, chiT2_raster_fm*lambdas[3]/lambdas[2]/lambdas[0]**2.0, ls = 'dashed', c = 'k', label='explicitly')
ylabel(r'$\chi_2/T^{\,2}$', labelpad = 0.1)
ax = subplot(111)
ax.set_ylim(0, 0.4)

### chi_3
figure(2)
plot(T_raster, chi_3_raster/T_raster*lambdas[3]/lambdas[2]**2.0*100, label = r'$\mu = %2.4f \, [MeV]$' %muval)
ylabel(r'$100 \chi_3/T$', labelpad = 0.1)
ax = subplot(111)
ax.set_ylim(-1e-3, 1e-3)

### chi_4
figure(3)
errorbar(chi4_lat[:,0], chi4_lat[:,1], chi4_lat[:,2], ls='', marker='s', ms=MS, mew=MEW, mec='k', c='k', label = 'WuBp data')
plot(T_raster, chi_4_raster*lambdas[3]/lambdas[2]**3.0, label = r'$\mu = %2.4f \, [MeV]$' %muval, ls='', marker='s')
ylabel(r'$\chi_4(T, 0)$', labelpad = 0.1)
ax = subplot(111)
ax.set_ylim(-0.01, 0.1)

for i in range(1, 4):
    figure(i)
    xlabel(r'$T \, [MeV]$')
    legend(loc = 'best', fontsize = 18, numpoints=3)
    ax = subplot(111)
    ax.set_xlim(0, 600)
    nice_ticks()
    savefig(model_type+'/'+ftype+'/pdfs/'+savenames[i]+model_type+'_numerical.pdf')


show()