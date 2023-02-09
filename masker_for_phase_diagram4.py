import numpy as np
import numpy.ma as ma
from scipy.interpolate import splrep, splev
import pickle
from pylab import *

from args_and_lambds import args_dic, lambdas_dic
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from args_and_lambds import args_dic, lambdas_dic

model_type = 'VRY_2'
ftype = args_dic['ftype'][model_type]

V_args = args_dic['V'][model_type]
f_args = args_dic['f'][model_type]
lambdas = lambdas_dic[model_type]

if model_type == 'G':
    fname = 'TD_gr_G_wmu0.p'
    fname2 = 'p_grid_G.p'
if model_type == 'no':
    fname = 'TD_gr_no.p'
if model_type == 'VRY_2':
    fname = model_type+'/'+ftype+'/TD_gr_VRY_2_wmu0.p'
    fname2 = model_type+'/'+ftype+'/p_grid_VRY_2.p'

print fname
TD_gr = pickle.load(open(fname, "rb"))
file.close(open(fname))

p_grid = pickle.load(open(fname2, "rb"))
file.close(open(fname2))
print 'p_grid: ', p_grid

TD_grid = TD_gr[0]
TD_grid = TD_scale(TD_grid, lambdas)[0]
phiPhi_grid = TD_gr[1]
TD_full = TD_gr[2]

phi0_pts = len(phiPhi_grid[:, 0, 0])
Phi1_pts = len(phiPhi_grid[0, :, 0])
T_help_ltp = np.zeros((phi0_pts, Phi1_pts))
T_help_htp = np.zeros((phi0_pts, Phi1_pts))
p_grid_masked = np.ma.asarray(p_grid)
#phi0_pts = TD_full['phi0_pts']
#Phi1_pts = TD_full['Phi1_pts']

fname = model_type+'/'+ftype+'/phase_contour_and_spinodals_'+model_type+'.p'
phase_contour_and_spinodals = pickle.load(open(fname, "rb"))
file.close(open(fname))

Tc_muT = phase_contour_and_spinodals['phase_contour']['Tc_mu_T']
sn_ltp_htp_up = phase_contour_and_spinodals['phase_contour']['sn_Phiphi']
spinodals_muT =  phase_contour_and_spinodals['spinodals']['mu_T']

Tc_muT_tck = splrep(Tc_muT[0, :], Tc_muT[1, :])
ltp_spinodal_muT_tck = splrep(spinodals_muT['ltp'][0, :], spinodals_muT['ltp'][1, :])
htp_spinodal_muT_tck = splrep(spinodals_muT['htp'][0, :], spinodals_muT['htp'][1, :])

#plot(Tc_muT[0, :]/988.9, Tc_muT[1, :]/111.5)
#show()

sT3pc_mu_ltp_tck = splrep(Tc_muT[0, :], sn_ltp_htp_up['ltp']['sT3'])
sT3pc_mu_htp_tck = splrep(Tc_muT[0, :], sn_ltp_htp_up['htp']['sT3'])
sT3pc_mu_up_tck  = splrep(Tc_muT[0, :], sn_ltp_htp_up['up']['sT3'])

#plot(Tc_muT[0, :], sn_ltp_htp_up['ltp']['sT3'])
#plot(Tc_muT[0, :], sn_ltp_htp_up['htp']['sT3'])
#plot(Tc_muT[0, :], sn_ltp_htp_up['up']['sT3'])
#show()

nT3pc_mu_ltp_tck = splrep(Tc_muT[0, :], sn_ltp_htp_up['ltp']['nT3'])
nT3pc_mu_htp_tck = splrep(Tc_muT[0, :], sn_ltp_htp_up['htp']['nT3'])
nT3pc_mu_up_tck  = splrep(Tc_muT[0, :],  sn_ltp_htp_up['up']['nT3'])


#nT3pc_mu_ltp_tck = splrep(Tc_muT[0, :], sn_ltp_htp['ltp']['nT3'])
#nT3pc_mu_htp_tck = splrep(Tc_muT[0, :], sn_ltp_htp['htp']['nT3'])

# from pylab import figure, show, plot, legend
# mu2 = np.linspace(Tc_muT[0, 0], Tc_muT[0, -1], 2000)
# mu_ltp2 = np.linspace(spinodals_muT['ltp'][0, 0], spinodals_muT['ltp'][0, -1], 2000)
# mu_htp2 = np.linspace(spinodals_muT['htp'][0, 0], spinodals_muT['htp'][0, -1], 2000)
# figure(1)
# plot(Tc_muT[0, :], Tc_muT[1, :])
# plot(mu2, splev(mu2, Tc_muT_tck), ls = 'dashed')
#
# plot(spinodals_muT['ltp'][0, :], spinodals_muT['ltp'][1, :])
# plot(mu_ltp2, splev(mu_ltp2, ltp_spinodal_muT_tck), ls = 'dashed')
#
# plot(spinodals_muT['htp'][0, :], spinodals_muT['htp'][1, :])
# plot(mu_htp2, splev(mu_ltp2, htp_spinodal_muT_tck), ls = 'dashed')
#
# figure(2)
# plot(Tc_muT[0, :], sn_ltp_htp_up['ltp']['sT3'])
# plot(mu2, splev(mu2, sT3pc_mu_ltp_tck), ls = 'dashed')
# plot(Tc_muT[0, :], sn_ltp_htp_up['htp']['sT3'])
# plot(mu2, splev(mu2, sT3pc_mu_htp_tck), ls = 'dashed')
# plot(Tc_muT[0, :], sn_ltp_htp_up['up']['sT3'])
# plot(mu2, splev(mu2, sT3pc_mu_up_tck), ls = 'dashed')
#
# show()


def get_stable_phases_for_PD(TD_grid, phiPhi_grid, p_grid, phase, save = 0):
    TD_grid_m = np.ma.asarray(TD_grid)
    phiPhi_grid_m = np.ma.asarray(phiPhi_grid)
    p_grid_m = np.ma.asarray(p_grid)
    CEP_appr_loc = Tc_muT[:, 0]
    print 'CEP_appr_loc: ', CEP_appr_loc

    phi0_pts = len(phiPhi_grid[:, 0, 0])
    Phi1_pts = len(phiPhi_grid[0, :, 0])
    print phi0_pts, Phi1_pts

    for i in range(0, phi0_pts):
        for j in range(0, Phi1_pts):
            T = TD_grid_m[i, j, 0]
            mu = TD_grid_m[i, j, 2]
            sT3 = TD_grid_m[i, j, 1]/T**3.0
            p = p_grid_m[i, j]
            pT4 = p_grid_m[i, j]/T**4.0
            nT3 = TD_grid_m[i, j, 3]/T**3.0
            Tproj_phasecont = splev(mu, Tc_muT_tck)

            if T > 1000.0 or mu > 2200.0 or T == 0 or p == 0:
                #pass
                TD_grid_m[i, j, :] = ma.masked
                phiPhi_grid_m[i, j, :] = ma.masked
                p_grid_m[i, j] = ma.masked
            else:
                if mu > CEP_appr_loc[0] and T < CEP_appr_loc[1]:
                    Tproj_ltpspinod = splev(mu, ltp_spinodal_muT_tck)
                    Tproj_htpspinod = splev(mu, htp_spinodal_muT_tck)
                    #print Tproj_htpspinod, Tproj_phasecont, Tproj_ltpspinod

                    if T > Tproj_htpspinod and T < Tproj_ltpspinod:
                        sT3pc_mu_ltp = splev(mu, sT3pc_mu_ltp_tck)
                        sT3pc_mu_htp = splev(mu, sT3pc_mu_htp_tck)
                        sT3pc_mu_up = splev(mu, sT3pc_mu_up_tck)

                        nT3pc_mu_ltp = splev(mu, nT3pc_mu_ltp_tck)
                        nT3pc_mu_htp = splev(mu, nT3pc_mu_htp_tck)
                        nT3pc_mu_up  = splev(mu, nT3pc_mu_up_tck)

                        if sT3 <= sT3pc_mu_up and T >= Tproj_phasecont and (phase=='ltp' or phase=='all'): ##condition for removal of ltp point
                        #if nT3 <= nT3pc_mu_up and T >= Tproj_phasecont and (phase=='ltp' or phase=='all'):
                            TD_grid_m[i, j, :] = ma.masked
                            phiPhi_grid_m[i, j, :] = ma.masked
                            p_grid_m[i, j] = ma.masked
                            print 'masking T = %2.4f, mu = %2.4f, sT3 = %2.4f' %(T, mu, sT3)

                        if sT3 >= sT3pc_mu_up and T <= Tproj_phasecont and (phase=='htp' or phase=='all'): ##condition for removal of htp point
                        #if nT3 >= nT3pc_mu_up and T <= Tproj_phasecont and (phase=='htp' or phase=='all'):
                            TD_grid_m[i, j, :] = ma.masked
                            phiPhi_grid_m[i, j, :] = ma.masked
                            p_grid_m[i, j] = ma.masked
                            print 'masking T = %2.4f, mu = %2.4f, sT3 = %2.4f' %(T, mu, sT3)
                        else:
                            print 'TD vals in sp: T = %2.4f, mu = %2.4f, sT3 = %2.4f' %(T, mu, sT3), sT3pc_mu_ltp, sT3pc_mu_up, sT3pc_mu_htp, Tproj_ltpspinod, Tproj_phasecont, Tproj_htpspinod

    if save:
        fname = model_type+'/'+ftype+'/TD_grid_masked_'+model_type+'.p'
        pickle.dump(TD_grid_m, open(fname, "wb"))
        file.close(open(fname))

        fname = model_type+'/'+ftype+'/phiPhi_grid_masked_'+model_type+'.p'
        pickle.dump(phiPhi_grid_m, open(fname, "wb"))
        file.close(open(fname))

        fname = model_type+'/'+ftype+'/p_grid_masked_'+model_type+'.p'
        pickle.dump(p_grid_m, open(fname, "wb"))
        file.close(open(fname))


    return [TD_grid_m, phiPhi_grid_m, p_grid_m]


get_stable_phases_for_PD(TD_grid, phiPhi_grid, p_grid, 'all', save = 1)
#p_ltp = get_stable_phases_for_PD(TD_grid, phiPhi_grid, p_grid, 'htp', save = 0)
#p_htp = get_stable_phases_for_PD(TD_grid, phiPhi_grid, p_grid, 'ltp', save = 0)

#for i in range(0, phi0_pts):
    #for j in range(0, Phi1_pts):
        #T_ltp  = TD_grid[i, j, 0]
        #mu_ltp = TD_grid[i, j, 2]
        #Tproj_phasecont_ltp = splev(mu_ltp, Tc_muT_tck)

        #T_htp  = p_htp[0][i, j, 0]
        #mu_htp = p_htp[0][i, j, 2]
        #Tproj_phasecont_htp = splev(mu_htp, Tc_muT_tck)

        #if mu_ltp > 988.9 and T_ltp <= Tproj_phasecont_ltp:
            #p_grid_masked[i,j] = p_ltp[2][i,j]
            #T_help_ltp[i, j] = 1.0
        ##if mu_htp > 988.9 and T_htp <= Tproj_phasecont_htp:
            ##p_grid_masked[i,j] = p_htp[2][i,j]
        #else:
            #p_grid_masked[i,j] = p_htp[2][i,j]
#fname = model_type+'/'+ftype+'/p_grid_masked_'+model_type+'.p'
#pickle.dump(p_grid_masked, open(fname, "wb"))
#file.close(open(fname))

##contour(p_ltp[0][:, :, 2]/988.9, p_ltp[0][:, :, 0]/111.5, p_ltp[2]*T_help_ltp, cmap = cm.jet, linewidths = 5, levels=np.linspace(p_ltp[2].min(), 0.02*p_ltp[2].max(), 400))
##scatter(p_ltp[0][:, :, 2]/988.9, p_ltp[0][:, :, 0]/111.5)
#contour(p_htp[0][:, :, 2]/988.9, p_htp[0][:, :, 0]/111.5, p_grid_masked, cmap = cm.jet, linewidths = 5, levels=np.linspace(p_grid_masked.min(), 0.02*p_grid_masked.max(), 400))
#plot(Tc_muT[0, :]/988.9, Tc_muT[1, :]/111.5, c='grey', lw=5)
#axis([0, 2, 0, 2])
#show()

print '\ndone: masker'