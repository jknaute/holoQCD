import numpy as np
from scipy.interpolate import splrep, splev
from scipy.integrate import quad

from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp, subplots, gcf, tight_layout, subplots_adjust, grid, title
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, LinearLocator
from matplotlib.transforms import Bbox
from nice_tcks import nice_ticks
from scipy.misc import derivative as deriv
from args_and_lambds import args_dic, lambdas_dic
from rasterizer import rasterize

import pickle
from Vtypes import Vs, dVs, dlogVs

def V(phi, *args):
    return Vs[args[0]](phi, *args[1])

def dV_dphi(phi, *args):
    return dVs[args[0]](phi, *args[1])

eos_lat = pickle.load(open('QG_latticedata_WuB.p', "rb"))
file.close(open('QG_latticedata_WuB.p'))
print eos_lat['T']
mu0_pT4_tck = splrep(eos_lat['T'], eos_lat['pT4'])
mu0_sT3_tck = splrep(eos_lat['T'], eos_lat['sT3'], s = 0.0)

chi2_lat = pickle.load(open('chi2_wubp.p', "rb"))
file.close(open('chi2_wubp.p'))

Tchi = chi2_lat[:, 0]
print Tchi
chi2hat = chi2_lat[:, 1]
mu0_chi2hat_tck = splrep(Tchi, chi2hat, s = 0.0)

def sT3_mu0(T, sT3_tck):
    return splev(T, sT3_tck)

def dsT3_dT_mu0(T, sT3_tck):
    return splev(T, sT3_tck, der = 1)

def chi2hat_mu0(T, chi2hat_tck):
    return splev(T, chi2hat_tck)

def dchi2hat_dT_mu0(T, chi2hat_tck):
    return splev(T, chi2hat_tck, der = 1)

model_type = 'G'
TDTA = pickle.load(open('TDTA_'+model_type+'.p', "rb"))
file.close(open('TDTA_'+model_type+'.p'))
TD_data_mu0 = TDTA[0]
phi0_raster = TDTA[1]
T_raster_model = TD_data_mu0[:, 0]
T_raster_model = np.compress(T_raster_model <= 510.0, T_raster_model)
cutoff_ind = len(T_raster_model)
TD_data_mu0 = TD_data_mu0[:cutoff_ind]
phi0_raster = phi0_raster[:cutoff_ind]

cutoff_ind_2 = len(np.compress(T_raster_model < 135.0, T_raster_model))
TD_data_mu0 = TD_data_mu0[:-cutoff_ind_2]
phi0_raster = phi0_raster[:-cutoff_ind_2]
T_raster_model = T_raster_model[:-cutoff_ind_2]
print T_raster_model

model_T_phiH_tck = splrep(phi0_raster, T_raster_model)
dTdphi_H_raster = splev(phi0_raster, model_T_phiH_tck, der = 1)

V_args = args_dic['V'][model_type]

def integrand(phi, V_args, mq):
    return V(phi, *V_args)/dV_dphi(phi, *V_args) + 12.0/(mq*phi)

def integral(phi, V_args, mq):
    return quad(integrand, 0, phi, args = (V_args, mq), limit = 400, epsrel = 1e-10, epsabs = 1e-10)[0]

mq = 2.0*V_args[1][1] - 12.0*V_args[1][0]**2.0

phi0_raster0 = np.linspace(0.001, 15, 2000)
#figure(10)
#plot(phi0_raster0, rasterize(V, phi0_raster0, *(V_args))[1]/rasterize(dV_dphi, phi0_raster0, *(V_args))[1])
#plot(phi0_raster0, 12.0/(mq*phi0_raster0))
#plot(phi0_raster0, rasterize(integrand, phi0_raster0, *(V_args, mq))[1])
#figure(11)
#plot(phi0_raster, dTdphi_H_raster)
#show()

integral_raster = np.zeros(len(phi0_raster))
V_part_raster = np.zeros(len(phi0_raster))
sigma_AA_raster = np.zeros(len(phi0_raster))
dsigma_dphiH_AA_raster = np.zeros(len(phi0_raster))
for i in range(0, len(phi0_raster)):
    phi_H = phi0_raster[i]
    integral_raster[i] = integral(phi_H, V_args, mq)
    V_part_raster[i] = (-V(phi_H, *V_args))**(1.0/2.0)/dV_dphi(phi_H, *V_args)*np.exp(2.0/3.0*integral_raster[i])*phi_H**(-8.0/mq)
    sigma_AA_raster[i] = (V(phi_H, *V_args)/-12.0)**(-3.0/2.0)
    dsigma_dphiH_AA_raster[i] = -3.0/2.0*(V(phi_H, *V_args)/-12.0)**(-5.0/2.0)*dV_dphi(phi_H, *V_args)/-12.0

TD_latpart = 1.0/(dsT3_dT_mu0(T_raster_model, mu0_sT3_tck)/chi2hat_mu0(T_raster_model, mu0_chi2hat_tck) - sT3_mu0(T_raster_model, mu0_sT3_tck)*dchi2hat_dT_mu0(T_raster_model, mu0_chi2hat_tck)/chi2hat_mu0(T_raster_model, mu0_chi2hat_tck)**2.0)
#f_AA_raster = TD_latpart*1.0/(dTdphi_H_raster)*V_part_raster
latAA_raster = dsigma_dphiH_AA_raster/chi2hat_mu0(T_raster_model, mu0_chi2hat_tck) - sigma_AA_raster*dchi2hat_dT_mu0(T_raster_model, mu0_chi2hat_tck)/chi2hat_mu0(T_raster_model, mu0_chi2hat_tck)**2.0*dTdphi_H_raster
f_AA_raster = 1.0/latAA_raster*V_part_raster

print integral_raster
print V_part_raster
print TD_latpart

chi2hat_sT3_ratio = chi2hat_mu0(T_raster_model, mu0_chi2hat_tck)/sT3_mu0(T_raster_model, mu0_sT3_tck)
figure(12)
plot(T_raster_model, chi2hat_mu0(T_raster_model, mu0_chi2hat_tck))
plot(T_raster_model, sT3_mu0(T_raster_model, mu0_sT3_tck))
# figure(13)
# plot(T_raster_model, chi2hat_sT3_ratio)
figure(14)
plot(phi0_raster, 1.0/chi2hat_sT3_ratio)

sT3_chi2hat_ratio = 1.0/chi2hat_sT3_ratio
sT3chi2hr_tck = splrep(phi0_raster, sT3_chi2hat_ratio, k = 3, s = 1)

plot(phi0_raster, splev(phi0_raster, sT3chi2hr_tck))

figure(15)
dsT3_chi2hat_ratio_dphi0 = splev(phi0_raster, sT3chi2hr_tck, der = 1)
plot(phi0_raster, dsT3_chi2hat_ratio_dphi0)


#plot(phi0_raster, sT3_mu0(T_raster_model, mu0_sT3_tck))
#plot(phi0_raster, dsT3_dT_mu0(T_raster_model, mu0_sT3_tck))

# figure(14)
# plot(eos_lat['T'], sT3_mu0(eos_lat['T'], mu0_sT3_tck))
# plot(eos_lat['T'], dsT3_dT_mu0(eos_lat['T'], mu0_sT3_tck))
# Tl2 = np.linspace(eos_lat['T'][0], eos_lat['T'][-1], 2000)
# plot(Tl2, sT3_mu0(Tl2, mu0_sT3_tck))
# figure(15)
# plot(Tchi, chi2hat)
# plot(Tchi, dchi2hat_dT_mu0(Tchi, mu0_chi2hat_tck))

print f_AA_raster
# figure(1)
# plot(T_raster_model, f_AA_raster)
# figure(2)
# plot(phi0_raster, f_AA_raster)

show()