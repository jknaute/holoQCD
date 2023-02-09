import numpy as np
import numpy.ma as ma
from scipy.interpolate import splrep, splev
import pickle

from args_and_lambds import args_dic, lambdas_dic
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from pylab import figure, show, legend, plot, scatter

model_type = 'G'
V_args = args_dic['V'][model_type]
f_args = args_dic['f'][model_type]
lambdas = lambdas_dic[model_type]

if model_type == 'G':
    fname = 'TD_gr_G_wmu0.p'
if model_type == 'no':
    fname = 'TD_gr_no.p'
print fname
TD_gr = pickle.load(open(fname, "rb"))
file.close(open(fname))

TD_grid = TD_gr[0]
TD_grid = TD_scale(TD_grid, lambdas)[0]
phiPhi_grid = TD_gr[1]
TD_full = TD_gr[2]

phi0_pts = TD_full['phi0_pts']
Phi1_pts = TD_full['Phi1_pts']

fname = 'phase_contour_and_spinodals_'+model_type+'_3.p'
phase_contour_and_spinodals = pickle.load(open(fname, "rb"))
file.close(open(fname))

print phase_contour_and_spinodals['phase_contour'].keys()
print phase_contour_and_spinodals['phase_contour']['sn_Phiphi']
print 4*'#'
Tc_muT = phase_contour_and_spinodals['phase_contour']['Tc_mu_T']
sn_Pp_ltp_htp = phase_contour_and_spinodals['phase_contour']['sn_Phiphi']
spinodals_muT = phase_contour_and_spinodals['spinodals']['mu_T']
sT3_ltp_spinodal = phase_contour_and_spinodals['spinodals']['sT3']['ltp']
sT3_htp_spinodal = phase_contour_and_spinodals['spinodals']['sT3']['htp']

Tc_muT_tck = splrep(Tc_muT[0, :], Tc_muT[1, :])
ltp_spinodal_muT_tck = splrep(spinodals_muT['ltp'][0, :], spinodals_muT['ltp'][1, :])
htp_spinodal_muT_tck = splrep(spinodals_muT['htp'][0, :], spinodals_muT['htp'][1, :])

sT3pc_mu_ltp_tck = splrep(Tc_muT[0, :], sn_Pp_ltp_htp['ltp']['sT3'])
sT3pc_mu_htp_tck = splrep(Tc_muT[0, :], sn_Pp_ltp_htp['htp']['sT3'])
sT3pc_mu_up_tck = splrep(Tc_muT[0, :], sn_Pp_ltp_htp['up']['sT3'])

sT3_ltp_spinodal_tck = splrep(Tc_muT[0, :], sT3_ltp_spinodal, s = 0.5)
sT3_htp_spinodal_tck = splrep(Tc_muT[0, :], sT3_htp_spinodal, s = 0.5)

# mu2 = np.linspace(Tc_muT[0, 0], Tc_muT[0, -1], 2000)
# figure(1)
# plot(Tc_muT[0, :], sT3_ltp_spinodal)
# plot(mu2, splev(mu2,sT3_ltp_spinodal_tck), ls = 'dashed')
# figure(2)
# plot(Tc_muT[0, :], sT3_htp_spinodal)
# plot(mu2, splev(mu2, sT3_htp_spinodal_tck), ls = 'dashed')
# show()

TD_grid_m = np.ma.asarray(TD_grid)
phiPhi_grid_m = np.ma.asarray(phiPhi_grid)

TD_grid_m_ltp = np.ma.asarray(TD_grid)
phiPhi_grid_m_ltp = np.ma.asarray(phiPhi_grid)

TD_grid_m_htp = np.ma.asarray(TD_grid)
phiPhi_grid_m_htp = np.ma.asarray(phiPhi_grid)

CEP_appr_loc = np.zeros(2)
CEP_appr_loc[:] = Tc_muT[:, 0]
CEP_appr_loc[0] = CEP_appr_loc[0] - 16 ## mu
CEP_appr_loc[1] = CEP_appr_loc[1] + 5 ## T
print CEP_appr_loc

sfac_h = 0.3 ## 1: from spinodal, 0: from T_c
sfac_l = 0.2 ## 1: from spinodal, 0: from T_c
for i in range(0, phi0_pts):
    for j in range(0, Phi1_pts):
        T = TD_grid_m[i, j, 0]
        mu = TD_grid_m[i, j, 2]
        sT3 = TD_grid_m[i, j, 1]/T**3.0
        if T > CEP_appr_loc[1] or mu < CEP_appr_loc[0] or mu > 1450 or T == 0: ##choose lower right part of phase
        # diagram starting from approximate CEP
            TD_grid_m_ltp[i, j, :] = ma.masked
            phiPhi_grid_m_ltp[i, j, :] = ma.masked
            TD_grid_m_htp[i, j, :] = ma.masked
            phiPhi_grid_m_htp[i, j, :] = ma.masked
        else:
            #if mu > CEP_appr_loc[0] and T < CEP_appr_loc[1]:
            Tproj_phasecont = splev(mu, Tc_muT_tck)
            Tproj_ltpspinod = splev(mu, ltp_spinodal_muT_tck)
            Tproj_htpspinod = splev(mu, htp_spinodal_muT_tck)
            sT3_spinodal_ltp = splev(mu, sT3_ltp_spinodal_tck)
            sT3_spinodal_htp = splev(mu, sT3_htp_spinodal_tck)
            sT3pc_mu_up = splev(mu, sT3pc_mu_up_tck)

            htp_Tshift = Tproj_phasecont - Tproj_htpspinod
            ltp_Tshift = Tproj_ltpspinod - Tproj_phasecont
            #print Tproj_htpspinod, Tproj_phasecont, Tproj_phasecont*1.02, Tproj_ltpspinod, T, sT3pc_mu_up, sT3, sT3_spinodal_htp#, htp_Tshift, ltp_Tshift

            if T >= Tproj_ltpspinod or T <= Tproj_htpspinod:
                TD_grid_m_ltp[i, j, :] = ma.masked
                phiPhi_grid_m_ltp[i, j, :] = ma.masked
            else:
                if sT3 >= sT3_spinodal_ltp or T >= Tproj_phasecont + sfac_l*ltp_Tshift:
                    TD_grid_m_ltp[i, j, :] = ma.masked
                    phiPhi_grid_m_ltp[i, j, :] = ma.masked

            if T >= Tproj_ltpspinod or T <= Tproj_htpspinod:
                TD_grid_m_htp[i, j, :] = ma.masked
                phiPhi_grid_m_htp[i, j, :] = ma.masked
            else:
                if T >= Tproj_phasecont:
                    if sT3 <= sT3pc_mu_up*1.02:
                        TD_grid_m_htp[i, j, :] = ma.masked
                        phiPhi_grid_m_htp[i, j, :] = ma.masked
                if T <= Tproj_phasecont*1.02:
                    if sT3 <= sT3_spinodal_htp:
                        TD_grid_m_htp[i, j, :] = ma.masked
                        phiPhi_grid_m_htp[i, j, :] = ma.masked
                if sT3 >= sT3pc_mu_up and T < Tproj_phasecont - sfac_h*htp_Tshift:
                    TD_grid_m_htp[i, j, :] = ma.masked
                    phiPhi_grid_m_htp[i, j, :] = ma.masked

b = np.zeros(1)
for i in range(0, phi0_pts):
    for j in range(0, Phi1_pts):
        #print np.round(np.ma.compressed(phiPhi_grid_m_htp[i, j, 1]), 2)  #, np.round(np.asarray(phiPhi_grid_m_ltp[i, j, 0], 2))
        ms = 'o'
        if i == 25:
            ms = '*'
        if i == 48:
            ms = 's'
        if i == 50:
            ms = 'd'

        figure(3)
        plot(phiPhi_grid_m_ltp[i, j, 1], phiPhi_grid_m_ltp[i, j, 0], color = 'green', marker = ms)
        plot(phiPhi_grid_m_htp[i, j, 1], phiPhi_grid_m_htp[i, j, 0], color = 'blue', marker = ms)
        figure(4)
        plot(TD_grid_m_ltp[i, j, 2], TD_grid_m_ltp[i, j, 0], color = 'green', marker = ms)
        plot(TD_grid_m_htp[i, j, 2], TD_grid_m_htp[i, j, 0], color = 'blue', marker = ms)

figure(4)
plot(Tc_muT[0, :], Tc_muT[1, :])
plot(spinodals_muT['ltp'][0, :], spinodals_muT['ltp'][1, :])
plot(spinodals_muT['htp'][0, :], spinodals_muT['htp'][1, :])

for phi_0 in phiPhi_grid_m_htp[:, :, 0]:
    print phi_0

#print np.ma.nonzero(phiPhi_grid_m_htp)
phiPhi_grid_htp_nz = phiPhi_grid_m_htp[np.ma.nonzero(phiPhi_grid_m_htp)]
phiPhi_grid_htp_nz_2 = np.vstack((phiPhi_grid_htp_nz[0::2], phiPhi_grid_htp_nz[1::2]))

phiPhi_grid_ltp_nz = phiPhi_grid_m_ltp[np.ma.nonzero(phiPhi_grid_m_ltp)]
phiPhi_grid_ltp_nz_2 = np.vstack((phiPhi_grid_ltp_nz[0::2], phiPhi_grid_ltp_nz[1::2]))

print phiPhi_grid_htp_nz_2
print phiPhi_grid_ltp_nz_2

phi0_raster_htp = np.zeros(len(phiPhi_grid_htp_nz_2[0, :]))
Phi1rmin_raster_htp = np.zeros((len(phiPhi_grid_htp_nz_2[0, :])))
Phi1rmax_raster_htp = np.zeros((len(phiPhi_grid_htp_nz_2[0, :])))

phi0_raster_ltp = np.zeros(len(phiPhi_grid_ltp_nz_2[0, :]))
Phi1rmin_raster_ltp = np.zeros((len(phiPhi_grid_ltp_nz_2[0, :])))
Phi1rmax_raster_ltp = np.zeros((len(phiPhi_grid_ltp_nz_2[0, :])))

### phi0, Phi1 for high T phase
i = 0
for k in range(0, len(phi0_raster_htp)):
    if k == 0:
        phi0_raster_htp[i] = phiPhi_grid_htp_nz_2[0, k]
        Phi1rmin_raster_htp[i] = phiPhi_grid_htp_nz_2[1, k]
        i += 1
    else:
        if phiPhi_grid_htp_nz_2[0, k] != phi0_raster_htp[i - 1]:
            print k
            phi0_raster_htp[i] = phiPhi_grid_htp_nz_2[0, k]
            Phi1rmin_raster_htp[i] = phiPhi_grid_htp_nz_2[1, k]
            Phi1rmax_raster_htp[i - 1] = phiPhi_grid_htp_nz_2[1, k - 1]
            i += 1
Phi1rmax_raster_htp[i] = phiPhi_grid_htp_nz_2[1, k]

### phi0, Phi1 for low T phase
i = 0
for k in range(0, len(phi0_raster_ltp)):
    if k == 0:
        phi0_raster_ltp[i] = phiPhi_grid_ltp_nz_2[0, k]
        Phi1rmin_raster_ltp[i] = phiPhi_grid_ltp_nz_2[1, k]
        i += 1
    else:
        if phiPhi_grid_ltp_nz_2[0, k] != phi0_raster_ltp[i - 1]:
            print k
            phi0_raster_ltp[i] = phiPhi_grid_ltp_nz_2[0, k]
            Phi1rmin_raster_ltp[i] = phiPhi_grid_ltp_nz_2[1, k]
            Phi1rmax_raster_ltp[i - 1] = phiPhi_grid_ltp_nz_2[1, k - 1]
            i += 1

print '\n *** i, k before problem: ', i, k
Phi1rmax_raster_ltp[i] = phiPhi_grid_ltp_nz_2[1, k]

#print phi0_raster_htp, Phi1rmin_raster_htp, Phi1rmax_raster_htp

phi0_raster_htp = phi0_raster_htp[np.nonzero(phi0_raster_htp)]
Phi1rmin_raster_htp = Phi1rmin_raster_htp[np.nonzero(Phi1rmin_raster_htp)]
Phi1rmax_raster_htp = Phi1rmax_raster_htp[np.nonzero(Phi1rmax_raster_htp)]

phi0_raster_ltp = phi0_raster_ltp[np.nonzero(phi0_raster_ltp)]
Phi1rmin_raster_ltp = Phi1rmin_raster_ltp[np.nonzero(Phi1rmin_raster_ltp)]
Phi1rmax_raster_ltp = Phi1rmax_raster_ltp[np.nonzero(Phi1rmax_raster_ltp)]
print 8*'#'

# for i in range(0, len(phi0_raster_htp)):
#     if Phi1rmin_raster_htp[i] == Phi1rmax_raster_htp[i]:
#          Phi1rmin_raster_htp[i] = 0.94*Phi1rmin_raster_htp[i]
Phi1rmin_raster_htp = 0.94*Phi1rmin_raster_htp

# for i in range(0, len(phi0_raster_ltp)):
#     if Phi1rmin_raster_ltp[i] == Phi1rmax_raster_ltp[i]:
#          Phi1rmin_raster_ltp[i] = 0.94*Phi1rmin_raster_ltp[i]
Phi1rmin_raster_ltp = 0.94*Phi1rmin_raster_ltp
#Phi1rmax_raster_ltp = 1.05*Phi1rmax_raster_ltp

print phi0_raster_htp
print Phi1rmin_raster_htp
print Phi1rmax_raster_htp
print 8*'#'
print phi0_raster_ltp
print Phi1rmin_raster_ltp
print Phi1rmax_raster_ltp

n_phi0_betw = 2
n_npts_ltp = (n_phi0_betw + 1)*len(phi0_raster_ltp) - (n_phi0_betw + 1)
n_npts_htp = (n_phi0_betw + 1)*len(phi0_raster_htp) - (n_phi0_betw + 1)

phi0Phi1r_new_ltp = np.zeros((3, n_npts_ltp + 1))
phi0Phi1r_new_htp = np.zeros((3, n_npts_htp + 1))

### new points for ltp:
for i in range(0, len(phi0_raster_ltp) - 1):
    for j in range(0, n_phi0_betw + 1):
        phi0Phi1r_new_ltp[0, i*(n_phi0_betw + 1) + j] = phi0_raster_ltp[i] + j/np.float(n_phi0_betw + 1)*(phi0_raster_ltp[i + 1] - phi0_raster_ltp[i])
        phi0Phi1r_new_ltp[1, i*(n_phi0_betw + 1) + j] = Phi1rmin_raster_ltp[i] + j/np.float(n_phi0_betw + 1)*(Phi1rmin_raster_ltp[i + 1] - Phi1rmin_raster_ltp[i])
        phi0Phi1r_new_ltp[2, i*(n_phi0_betw + 1) + j] = Phi1rmax_raster_ltp[i] + j/np.float(n_phi0_betw + 1)*(Phi1rmax_raster_ltp[i + 1] - Phi1rmax_raster_ltp[i])
phi0Phi1r_new_ltp[:, -1] = np.array([phi0_raster_ltp[-1], Phi1rmin_raster_ltp[-1], Phi1rmax_raster_ltp[-1]])

### new points for htp:
for i in range(0, len(phi0_raster_htp) - 1):
    for j in range(0, n_phi0_betw + 1):
        phi0Phi1r_new_htp[0, i*(n_phi0_betw + 1) + j] = phi0_raster_htp[i] + j/np.float(n_phi0_betw + 1)*(phi0_raster_htp[i + 1] - phi0_raster_htp[i])
        phi0Phi1r_new_htp[1, i*(n_phi0_betw + 1) + j] = Phi1rmin_raster_htp[i] + j/np.float(n_phi0_betw + 1)*(Phi1rmin_raster_htp[i + 1] - Phi1rmin_raster_htp[i])
        phi0Phi1r_new_htp[2, i*(n_phi0_betw + 1) + j] = Phi1rmax_raster_htp[i] + j/np.float(n_phi0_betw + 1)*(Phi1rmax_raster_htp[i + 1] - Phi1rmax_raster_htp[i])
phi0Phi1r_new_htp[:, -1] = np.array([phi0_raster_htp[-1], Phi1rmin_raster_htp[-1], Phi1rmax_raster_htp[-1]])

#phi0Phi1r_new_ltp[1, :] = np.ones(len(phi0Phi1r_new_ltp[1, :]))*phi0Phi1r_new_ltp[1, 0]
#phi0Phi1r_new_ltp[2, :] = np.ones(len(phi0Phi1r_new_ltp[2, :]))*phi0Phi1r_new_ltp[2, 0]
for i in range(0, len(phi0Phi1r_new_ltp[0, :])):
    if phi0Phi1r_new_ltp[2, i] < phi0Phi1r_new_ltp[2, 0]:
        #print i, phi0Phi1r_new_ltp[2, 0], phi0Phi1r_new_ltp[2, i]
        print 'changing:', phi0Phi1r_new_ltp[1, i], phi0Phi1r_new_ltp[2, i]
        phi0Phi1r_new_ltp[1, i] = (phi0Phi1r_new_ltp[1, i] + phi0Phi1r_new_ltp[1, 0])/2.0
        phi0Phi1r_new_ltp[2, i] = (phi0Phi1r_new_ltp[2, i] + phi0Phi1r_new_ltp[2, 0])/2.0
        print phi0Phi1r_new_ltp[1, i], phi0Phi1r_new_ltp[2, i]

phi0Phi1r_new_ltp[1, :] = np.clip(phi0Phi1r_new_ltp[1, :], phi0Phi1r_new_ltp[1, -1], phi0Phi1r_new_ltp[1, 0])

def htp_leftmost_pt_inserter(phi0Phi1r_new_htp):
    nhtp_pts = np.zeros(3)
    nhtp_pts[0] =  phi0Phi1r_new_htp[0, -1] + np.abs(phi0Phi1r_new_htp[0, -1] - phi0Phi1r_new_htp[0, -2])
    nhtp_pts[1:] =  phi0Phi1r_new_htp[1:, -1] - np.abs(phi0Phi1r_new_htp[1:, -1] - phi0Phi1r_new_htp[1:, -2])
    print 'nhtp =', nhtp_pts
    phi0Phi1r_new_htp = np.insert(phi0Phi1r_new_htp, len(phi0Phi1r_new_htp[0,:]), nhtp_pts, axis = 1)
    return phi0Phi1r_new_htp

insert_leftmost_htp_pt = 1
if insert_leftmost_htp_pt:
    phi0Phi1r_new_htp = htp_leftmost_pt_inserter(phi0Phi1r_new_htp)
    phi0Phi1r_new_htp = htp_leftmost_pt_inserter(phi0Phi1r_new_htp)
    phi0Phi1r_new_htp = htp_leftmost_pt_inserter(phi0Phi1r_new_htp)

phi0Phi1r_new_htp[1, -5:] = 0.96*phi0Phi1r_new_htp[1, -5:]

phi0Phi1r_new_htp[2, :] = 1.01*phi0Phi1r_new_htp[2, :]

print phi0Phi1r_new_ltp
print phi0Phi1r_new_htp
#print buh
ltp_newpts = phi0Phi1r_new_ltp
htp_newpts = phi0Phi1r_new_htp

print ltp_newpts
print htp_newpts

figure(3)
for i in range(0, len(phi0Phi1r_new_ltp[0, :])):
    Phi1min = phi0Phi1r_new_ltp[1, i]
    Phi1max = phi0Phi1r_new_ltp[2, i]
    phi0 = phi0Phi1r_new_ltp[0, i]
    plot(np.array([Phi1min, Phi1max]), np.array([phi0, phi0]), color = 'green')
for i in range(0, len(phi0Phi1r_new_htp[0, :])):
    Phi1min = phi0Phi1r_new_htp[1, i]
    Phi1max = phi0Phi1r_new_htp[2, i]
    phi0 = phi0Phi1r_new_htp[0, i]
    plot(np.array([Phi1min, Phi1max]), np.array([phi0, phi0]), color = 'blue')

for ph in ['ltp', 'htp', 'up']:
    plot(sn_Pp_ltp_htp[ph]['Phi'], sn_Pp_ltp_htp[ph]['phi'])

## (phi_0, Phi_1r)(p_c(T, u)) ltp
ltp_phiPhi_tck = splrep(sn_Pp_ltp_htp['ltp']['phi'], sn_Pp_ltp_htp['ltp']['Phi'])
phi0_Tc_newpts_ltp = phi0Phi1r_new_ltp[0, :]
Phi1r_Tc_newpts_ltp = splev(phi0_Tc_newpts_ltp, ltp_phiPhi_tck)
ltp_pc_onnewpts = np.vstack((phi0_Tc_newpts_ltp, Phi1r_Tc_newpts_ltp)) # <===

## (phi_0, Phi_1r)(p_c(T, u)) htp
print sn_Pp_ltp_htp['htp']['phi']
htp_phiPhi_tck = splrep(sn_Pp_ltp_htp['htp']['phi'][::-1], sn_Pp_ltp_htp['htp']['Phi'][::-1])
htp_Phiphi_tck = splrep(sn_Pp_ltp_htp['htp']['Phi'], sn_Pp_ltp_htp['htp']['phi'])

Phi1r_Tc_htp_max = phi0Phi1r_new_htp[2, :].max()
phi0_Tc_htp_min = splev(Phi1r_Tc_htp_max, htp_Phiphi_tck)
print Phi1r_Tc_htp_max, phi0_Tc_htp_min

phi0_Tc_newpts_htp = np.compress(phi0Phi1r_new_htp[0, :] >= phi0_Tc_htp_min, phi0Phi1r_new_htp[0, :])
Phi1r_Tc_newpts_htp = splev(phi0_Tc_newpts_htp, htp_phiPhi_tck)

htp_pc_onnewpts = np.vstack((phi0_Tc_newpts_htp, Phi1r_Tc_newpts_htp)) # <===

plot(Phi1r_Tc_newpts_ltp, phi0_Tc_newpts_ltp)
plot(Phi1r_Tc_newpts_htp, phi0_Tc_newpts_htp)

###
logpc_pts = np.zeros(len(Tc_muT[0, :]))
k = 0
for mu_val in Tc_muT[0, :]:
    print mu_val, phase_contour_and_spinodals['p_Tc_data'][mu_val]['p']
    #logpc_pts[k] = np.log(phase_contour_and_spinodals['p_Tc_data'][mu_val]['p_scaled'])
    logpc_pts[k] = np.log(phase_contour_and_spinodals['p_Tc_data'][mu_val]['p'])
    k+=1

logpc_ltp_tck = splrep(sn_Pp_ltp_htp['ltp']['phi'], logpc_pts)
logpc_htp_tck = splrep(sn_Pp_ltp_htp['htp']['phi'][::-1], logpc_pts[::-1])

ltp_pc_newpts = np.exp(splev(phi0_Tc_newpts_ltp, logpc_ltp_tck))
htp_pc_newpts = np.exp(splev(phi0_Tc_newpts_htp, logpc_htp_tck))

figure(5)
plot(sn_Pp_ltp_htp['ltp']['phi'], logpc_pts, color = 'green')
plot(phi0_Tc_newpts_ltp, np.log(ltp_pc_newpts), color = 'green', ls = 'dashed')
plot(sn_Pp_ltp_htp['htp']['phi'][::-1], logpc_pts[::-1], color = 'blue')
plot(phi0_Tc_newpts_htp, np.log(htp_pc_newpts), color = 'blue', ls = 'dashed')

ltp_newpts_pcline = np.vstack((ltp_pc_onnewpts, ltp_pc_newpts))
htp_newpts_pcline = np.vstack((htp_pc_onnewpts, htp_pc_newpts))

print 10*'#'
print ltp_newpts_pcline
print htp_newpts_pcline

htp_pts_no_pc = np.compress(phi0Phi1r_new_htp[0, :] < phi0_Tc_htp_min, phi0Phi1r_new_htp[0, :])
htp_pts_no_pc = np.vstack((htp_pts_no_pc, np.zeros(len(htp_pts_no_pc)), np.zeros(len(htp_pts_no_pc))))
htp_newpts_pcline = np.hstack((htp_pts_no_pc, htp_newpts_pcline))

print ltp_newpts_pcline
print htp_newpts_pcline

pclines = {'ltp': ltp_newpts_pcline, 'htp':htp_newpts_pcline}
#print phase_contour_and_spinodals['p_Tc_data'].keys()#['p_scaled']
#print Tc_muT

#fname = 'TD_grid_masked_G.p'
#pickle.dump(TD_grid_m, open(fname, "wb"))
#file.close(open(fname))

fname = 'phiPhi_range_htp_16.p'
pickle.dump(htp_newpts, open(fname, "wb"))
file.close(open(fname))
fname = 'phiPhi_range_ltp_16.p'
pickle.dump(ltp_newpts, open(fname, "wb"))
file.close(open(fname))

fname = 'pc_lines_G_16.p'
pickle.dump(pclines, open(fname, "wb"))
file.close(open(fname))


show()