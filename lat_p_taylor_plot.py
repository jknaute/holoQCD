import numpy as np
from scipy.interpolate import splrep, splev

from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp, subplots, gcf, tight_layout, subplots_adjust, grid
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

chi2_lat = pickle.load(open('chi2_wubp.p', "rb"))
file.close(open('chi2_wubp.p'))

chi4_lat = pickle.load(open('chi4_wubp.p', "rb"))
file.close(open('chi4_wubp.p'))

T = chi2_lat[:, 0]
chi2hat = chi2_lat[:, 1]
T = T[:-2]
chi2hat = chi2hat[:-2]

Tchi4 = chi4_lat[:,0]
chi4 = chi4_lat[:, 1]
print T, Tchi4[:-2]#, chi2hat
#print buh

eos_vals_chooser = np.zeros(len(T), dtype = int)
for i in range(0, len(T)):
    eos_vals_chooser[i] = np.where(eos_lat['T'] == T[i])[0][0]
for key in eos_lat.keys():
    eos_lat.update({key:eos_lat[key][eos_vals_chooser]})
print eos_lat['T']

chi4 = splev(T, splrep(Tchi4, chi4))

mu_levels = np.arange(0, 1600.6, 400)
chi2 = T**2.0*chi2hat

def func(T, tck):
    return splev(T, tck)

dchi2hat_dT = splev(T, splrep(T, chi2hat), der = 1)
dchi2_dT = 2.0*T*chi2hat + T**2.0*dchi2hat_dT
#dchi2_dT = splev(T, splrep(T, chi2), der = 1)

dchi4_dT = splev(T, splrep(T, chi4), der = 1)

figure(10)
plot(T, chi2hat, label = r'$\hat \chi_2(T)$')
plot(T, chi4, label = r'$\chi_4(T)$')
xlabel(r'$T\, [MeV]$')
legend(loc = 'best', fontsize = 16)
nice_ticks()

plot_nnlo = 0
for mu in mu_levels:
    pT4_nlo = 1.0/2.0*mu**2.0/T**2.0*chi2hat
    s_nlo = 1.0/2.0*mu**2.0*dchi2_dT
    sT3_nlo = s_nlo/T**3.0
    nT3_nlo = mu/T*chi2hat

    pT4_nnlo = 1.0/24.0*mu**4.0/T**4.0*chi4
    s_nnlo = 1.0/24.0*mu**4.0*dchi4_dT
    sT3_nnlo = s_nnlo/T**3.0
    nT3_nnlo = 1.0/6.0*mu**3.0/T**3.0*chi4

    pT4_fm = pT4_nlo
    sT3_fm = sT3_nlo
    nT3_fm = nT3_nlo

    if plot_nnlo:
        pT4_fm = pT4_nlo + pT4_nnlo
        # sT3_fm = sT3_nlo + sT3_nnlo
        nT3_fm = nT3_nlo + nT3_nnlo

    figure(1)
    plot(T, eos_lat['pT4'] + pT4_fm, label = r'$\mu = %d \, MeV$' %mu)

    figure(2)
    fig2, = plot(T, eos_lat['sT3'] + sT3_fm, label = r'$\mu = %d \, MeV$' %mu)
    clr = fig2.get_color()
    if mu > 0:
        figure(3)
        plot(T, pT4_nlo, label = r'$\mu = %d \, MeV$' %mu, color = clr)
        if plot_nnlo:
            plot(T, pT4_nnlo, color = clr, ls = 'dashed')

        figure(4)
        plot(T, sT3_fm, label = r'$\mu = %d \, MeV$' %mu, color = clr)

        figure(5)
        plot(T, nT3_fm, label = r'$\mu = %d \, MeV$' %mu, color = clr)
        #if plot_nnlo:
        #    plot(T, nT3_fm, color = clr, ls = 'dashed')



savenames = {1:'p_lo_nlo', 2:'s_lo_nlo', 3:'p_nlo', 4:'s_nlo', 5:'n_nlo'}

figure(1)
#ylabel(r'$p(T, 0)/T^{\,4} + \frac{1}{2} \mu^2/T^{\,2} \hat\chi_2$')
if not plot_nnlo:
    ylabel(r'$(p(T, 0) + p_{nlo}(T, \mu))/T^{\,4}$')
else:
    ylabel(r'$(p(T, 0) + p_{nlo}(T, \mu) + p_{nnlo}(T, \mu))/T^{\,4}$')
axis([120, 500, 0, 10])

figure(2)
#ylabel(r'$s(T, 0)/T^{\,3} + \frac{1}{2} \mu^2 \partial(\hat\chi_2 T^{\,2})/\partial T$')
if not plot_nnlo:
    ylabel(r'$(s(T, 0) + s_{nlo}(T, \mu))/T^{\,3}$')
else:
    ylabel(r'$(s(T, 0) + s_{nlo}(T, \mu) + s_{nnlo}(T, \mu))/T^{\,3}$')
axis([120, 500, 0, 60])

figure(3)
#ylabel(r'$\frac{1}{2} \mu^2/T^{\,2} \hat\chi_2$')
if not plot_nnlo:
    ylabel(r'$p_{nlo}(T, \mu)/T^{\,4}$')
else:
    ylabel(r'$p_{nlo}(T, \mu)/T^{\,4}, \, p_{nnlo}(T, \mu)/T^{\,4}$')
plot(T, eos_lat['pT4'], color = 'b', label = r'$p(T,0)/T^{\,4}$')
axis([120, 500, 0, 10])

figure(4)
#ylabel(r'$\frac{1}{2} \mu^2 \partial(\hat\chi_2 T^{\,2})/\partial T$')
if not plot_nnlo:
    ylabel(r'$s_{nlo}(T, \mu)/T^{\,3}$')
else:
    ylabel(r'$s_{nlo}(T, \mu)/T^{\,3}, \, s_{nnlo}(T, \mu)/T^{\,3}$')
plot(T, eos_lat['sT3'], color = 'b', label = r'$s(T,0)/T^{\,3}$')
axis([120, 500, 0, 60])

figure(5)
if not plot_nnlo:
    ylabel(r'$n_{nlo}(T, \mu)/T^{\,3}$')
else:
    #ylabel(r'$n_{nlo}(T, \mu)/T^{\,3}, n_{nnlo}(T, \mu)/T^{\,3}$')
    ylabel(r'$(n_{nlo}(T, \mu) + n_{nnlo}(T, \mu))/T^{\,3}$')
axis([120, 500, 0, 8])

for i in range(1,6):
    figure(i)
    xlabel(r'$T\, [MeV]$')
    nice_ticks()
    if i == 1 and plot_nnlo:
        legend(fontsize = 16, loc = 'best', ncol = 2)
    else:
        legend(fontsize = 16, loc = 'best')
    #legend(loc = 'best')
    # savename = savenames[i]
    # if plot_nnlo:
    #     savename = savename+'_nnlo'
    # savefig(savename+'.pdf')


show()