import numpy as np
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from scipy.interpolate import splrep, splev
from args_and_lambds import args_dic, lambdas_dic

import pickle
from amoba import amoeba

from p_calcer4 import p_calc_line, p_calc_Tlvl, p_PT_calc, p_calc_mulvl
from more_TD import e_and_I, vsq, chi2_calc

from pylab import figure, plot, show, legend, scatter



#####################
model_type = 'VRY_4'
ftype = args_dic['ftype'][model_type]
V_args = args_dic['V'][model_type]
f_args = args_dic['f'][model_type]
lambdas = lambdas_dic[model_type]

if model_type == 'G':
    fname = 'TD_gr_G_wmu0.p'
if model_type == 'no':
    fname = 'TD_gr_no.p'
if model_type == 'VRY_2' or model_type == 'VRY_4':
    fname = model_type+'/'+ftype+'/TD_gr_'+model_type+'_wmu0.p'

print fname
TD_gr = pickle.load(open(fname, "rb"))

TD_grid = TD_gr[0]
TD_grid = TD_scale(TD_grid, lambdas)[0]
phiPhi_grid = TD_gr[1]
TD_full = TD_gr[2]

phi0_pts = TD_full['phi0_pts']
Phi1_pts = TD_full['Phi1_pts']

lphi0_raster = np.log(TD_full['phi0_raster'])
Phi1r_raster = phiPhi_grid[20, :][:, 1]

print lphi0_raster
print Phi1r_raster
print TD_grid[:, :, 0]

#### contours:
if model_type == 'G':
    fname = 'T_mu_contours_G.p'
    tmc = pickle.load(open(fname, "rb"))
    file.close(open(fname))

    # fname2 = 'T_mu_contours_G_mu_792.p'
    # tmc2 = pickle.load(open(fname2, "rb"))
    # file.close(open(fname))

    fname = 'TDTA_G.p'
    TDTA = pickle.load(open(fname, "rb"))
    file.close(open(fname))

elif model_type == 'no':
    fname = 'T_mu_contours_no.p'
    tmc = tmc = pickle.load(open(fname, "rb"))

    fname = 'TDTA_no.p'
    TDTA = pickle.load(open(fname, "rb"))
    file.close(open(fname))
elif model_type == 'VRY_2':
    fname = model_type+'/'+ftype+'/T_mu_contours_VRY_2_all.p'
    tmc = pickle.load(open(fname, "rb"))
    # fname2 = 'T_mu_contours_VRY_2_moremu.p'
    # tmc2 = pickle.load(open(fname2, "rb"))
    # tmc['mu'].update(tmc2['mu'])
    print tmc['mu'].keys()
    # pickle.dump(tmc, open('T_mu_contours_VRY_2.p', "wb"))
    # file.close(open('T_mu_contours_VRY_2.p'))

    fname = model_type+'/'+ftype+'/TDTA_VRY_2.p'
    TDTA = pickle.load(open(fname, "rb"))
    file.close(open(fname))
elif model_type == 'VRY_4':
    fname = model_type+'/'+ftype+'/T_mu_contours_'+model_type+'.p'
    tmc = pickle.load(open(fname, "rb"))

    fname = model_type+'/'+ftype+'/TDTA_'+model_type+'.p'
    TDTA = pickle.load(open(fname, "rb"))
    file.close(open(fname))


T_levels = np.sort(tmc['T'].keys())
mu_levels = np.sort(tmc['mu'].keys())

# print tmc2['mu']
# tmc['mu'].update(tmc2['mu'])
# print tmc['mu'].keys()
# pickle.dump(tmc, open('T_mu_contours_G.p', "wb"))
# file.close(open('T_mu_contours_G.p'))

print 'T_levels: ' , T_levels
print 'mu_levels: ', mu_levels


### thermodynamics on T-axis
TD_Tax = TDTA[0]
phi0_raster = TDTA[1]
Phiphi_T0 = np.vstack((np.zeros(len(phi0_raster)), phi0_raster))
p_Tax = p_calc_line([TD_Tax[0,0], 0], [TD_Tax[-1,0],0], TD_Tax, Phiphi_T0)
print 'p_Tax =', p_Tax

p_rasters_const_T = {}
p_rasters_const_mu = {}

### pressure along const T levels
for i in range(0, len(T_levels)):
    lvl = T_levels[i]
    acc_cntrTD = tmc['T'][lvl]

    TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]
    #TD_slice_scaled = acc_cntrTD[1]
    phiPhi_slice = acc_cntrTD[0]

    p_raster = p_calc_Tlvl(p_Tax, TD_slice_scaled, phiPhi_slice, lvl)
    print 'p_raster = ', p_raster
    p_rasters_const_T.update({lvl:p_raster})
    #plot(p_raster[2], p_raster[0])
    #show()

### pressure along const mu levels
#for i in range(0, len(mu_levels))[:-1]:
for i in range(0, len(mu_levels)):
    lvl = mu_levels[i]
    acc_cntrTD = tmc['mu'][lvl]

    TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]

    phiPhi_slice = acc_cntrTD[0]
    if lvl == 0:
        p_raster = [p_Tax[0],p_Tax[0]/p_Tax[1]**4.0, p_Tax[1], TD_Tax, Phiphi_T0]
    else:
        T_i = 55 # 50
        T_f = TD_slice_scaled[0, 0]
        print 'T_i, T_f:', T_i, T_f
        if T_i not in T_levels:
            print 'WARNING: T_i must match any of the T levels'

        p_Tlvl = p_rasters_const_T[T_i]
        p_raster = p_calc_mulvl([p_Tlvl[0], p_Tlvl[2]], TD_slice_scaled, phiPhi_slice, lvl, T_i, T_f)
        print 'p_list = ', p_raster[0], len(p_raster)
        print p_raster[-1]
        print len(p_raster)

    p_rasters_const_mu.update({lvl:p_raster})

### calculate phase contour
Tc_muT = np.zeros((2, len(mu_levels)))
muT_ltp_spinodal = np.zeros((2, len(mu_levels)))
muT_htp_spinodal = np.zeros((2, len(mu_levels)))

phiPhi_Tc = np.zeros((2, len(mu_levels)))
Phiphi_ltp_spinodal = np.zeros((2, len(mu_levels)))
Phiphi_htp_spinodal = np.zeros((2, len(mu_levels)))

sT3_ltp_spinodal = np.zeros(len(mu_levels))
sT3_htp_spinodal = np.zeros(len(mu_levels))

phasetrans = {}
#for k in range(0, len(mu_levels) - 1):
for k in range(0, len(mu_levels)):
    mu = mu_levels[k]
    print '\nmu = ', mu
    p_list = p_rasters_const_mu[mu]
    #print len(p_list)
    TD_n = p_list[3]
    Phiphi_n = p_list[4]

    tdv_raster = p_list[2]
    p_raster = p_list[0]
    ps_raster = p_list[0]/p_list[2]**4.0
    #print mu
    print 'tdv_raster, ps_raster: ', tdv_raster, ps_raster
    p_PTc = p_PT_calc(p_raster, ps_raster, tdv_raster)
    print 4*'#'
    print 'p_PTc: ', p_PTc
    print 4*'#'
    if p_PTc[0] == '1st':
        Tc_muT[:, k] = np.array([mu, p_PTc[3]])
        Tc = p_PTc[3]
        inds_Tc = p_PTc[4]
        inds_spinds = p_PTc[5]

        phasetrans.update({mu:{'type':p_PTc[0], 'p':p_PTc[1], 'p_scaled':p_PTc[2], 'inds':inds_Tc, 'inds_ltp_htp':inds_spinds, 'Tc_mu_T':Tc_muT[:, k]}})
        muT_ltp_spinodal[:, k] = np.array([mu, tdv_raster[inds_spinds[0]]])
        muT_htp_spinodal[:, k] = np.array([mu, tdv_raster[inds_spinds[1]]])
        sT3_ltp_spinodal[k] = np.array([TD_n[inds_spinds[0], 1]/tdv_raster[inds_spinds[0]]**3.0])
        sT3_htp_spinodal[k] = np.array([TD_n[inds_spinds[1], 1]/tdv_raster[inds_spinds[1]]**3.0])
        Phiphi_ltp_spinodal[:, k] = Phiphi_n[:, inds_spinds[0]]
        Phiphi_htp_spinodal[:, k] = Phiphi_n[:, inds_spinds[1]]
    else:
        phasetrans.update({mu:{'type':p_PTc[0]}})
        muT_ltp_spinodal[:, k] = 0
        muT_htp_spinodal[:, k] = 0
        sT3_ltp_spinodal[k] = 0
        sT3_htp_spinodal[k] = 0
        Phiphi_ltp_spinodal[:, k] = 0
        Phiphi_htp_spinodal[:, k] = 0
print 'Tc_muT: ', Tc_muT


def val_search_at_Tc(val, data):
    tck = data[0]
    T_c = data[1]
    return -(splev(val, tck) - T_c)**2.0

def get_s_and_n_ltphtp_Tc(p_rasters_const_mu, phasetrans, Tc_muT):
    sn_Phiphi_Tc = {'ltp':{'sT3':np.zeros(len(Tc_muT[0, :])), 'nT3':np.zeros(len(Tc_muT[0, :])), 'Phi':np.zeros(len(Tc_muT[0, :])), 'phi':np.zeros(len(Tc_muT[0, :]))},
                    'htp':{'sT3':np.zeros(len(Tc_muT[0, :])), 'nT3':np.zeros(len(Tc_muT[0, :])), 'Phi':np.zeros(len(Tc_muT[0, :])), 'phi':np.zeros(len(Tc_muT[0, :]))},
                    'up':{'sT3':np.zeros(len(Tc_muT[0, :])), 'nT3':np.zeros(len(Tc_muT[0, :])), 'Phi':np.zeros(len(Tc_muT[0, :])), 'phi':np.zeros(len(Tc_muT[0, :]))}}

    mu_vals = Tc_muT[0, :]
    for i in range(0, len(mu_vals)):
        mu = mu_vals[i]
        #if phasetrans[mu]['type'] == '1st':
        T_raster = p_rasters_const_mu[mu][3][:, 0]
        s_raster = p_rasters_const_mu[mu][3][:, 1]
        n_raster = p_rasters_const_mu[mu][3][:, 3]
        sT3 = s_raster/T_raster**3.0
        nT3 = n_raster/T_raster**3.0
        print '\nT_raster: ', T_raster, len(T_raster)

        Phi1_raster = p_rasters_const_mu[mu][4][0, :]
        phi0_raster = p_rasters_const_mu[mu][4][1, :]
        #print Phi1_raster, phi0_raster

        sn_Phiphi_rasters = {'sT3':sT3, 'nT3':nT3, 'Phi':Phi1_raster, 'phi':phi0_raster}

        Tc = phasetrans[mu]['Tc_mu_T'][1]
        print 'mu_c =', mu, 'T_c =', Tc

        ltp_maxind = phasetrans[mu]['inds_ltp_htp'][0]
        htp_minind = phasetrans[mu]['inds_ltp_htp'][1] + 1
        ltp_pt_ind = phasetrans[mu]['inds'][0]
        htp_pt_ind = phasetrans[mu]['inds'][1]

        #ltp_lind = np.amax([ltp_pt_ind - 3, 0])
        #htp_hind = np.amin([htp_pt_ind + 4, len(T_raster)])
        ltp_lind = np.amin([ltp_maxind - 4, ltp_pt_ind - 2])
        htp_hind = np.amax([htp_minind + 5, htp_pt_ind + 3])
        ltp_maxind = np.amin([ltp_pt_ind + 6, ltp_maxind])
        print 'ltp_lind =', ltp_lind, 'ltp_pt_ind =', ltp_pt_ind, 'htp_pt_ind =', htp_pt_ind, 'htp_hind =', htp_hind, 'ltp_maxind =', ltp_maxind, 'htp_minind =', htp_minind

        #sn_Phiphi_tcks = {'ltp':{}, 'htp':{}, 'up':{}}
        cut_inds =  {'ltp':[ltp_lind, ltp_maxind + 1], 'htp':[htp_minind, htp_hind], 'up':[ltp_maxind, htp_minind]}
        Tr2s = {'ltp':np.linspace(T_raster[ltp_lind], T_raster[ltp_maxind], 200), 'htp':np.linspace(T_raster[htp_minind], T_raster[htp_hind], 200), 'up':np.linspace(T_raster[htp_minind], T_raster[ltp_maxind], 200)[::-1]}
        j = 0
        for ph in ['ltp', 'htp', 'up']:
            for qty in sn_Phiphi_rasters.keys():
                print '\nph, qty: ', ph, qty
                k_s = 3
                if ph == 'up':
                    if cut_inds[ph][1] - cut_inds[ph][0] <= 4:
                        k_s = 1
                    elif cut_inds[ph][1] - cut_inds[ph][0] > 5:
                        cut_inds[ph][0] += 1
                        #cut_inds[ph][1] -= 1
                if ph == 'ltp' and qty == 'Phi' and cut_inds[ph][0] > 0:
                     cut_inds[ph][0] -= 1
                     print np.amax([cut_inds[ph][0], 0])
                     cut_inds[ph][1] -= 1
                print 'cut_inds[ph][0], cut_inds[ph][1]: ', cut_inds[ph][0], cut_inds[ph][1]
                qty_raster = sn_Phiphi_rasters[qty][cut_inds[ph][0]:cut_inds[ph][1]]
                T_raster_c = T_raster[cut_inds[ph][0]:cut_inds[ph][1]]

                print 'T_raster_c: ', T_raster_c
                print 'qty_raster: ', qty_raster
                #if len(T_raster_c) == 0:
                    #print '***problematic qty: ', mu, ph, qty
                    #continue
                if qty_raster[-1] < qty_raster[0]:
                    qty_raster = qty_raster[::-1]
                    T_raster_c = T_raster_c[::-1]

                x = 0
                der_cutoff = 5.0
                #if len(qty_raster) <= 3:
                    #k_s = 1
                try:
                    print 'k_s = ', k_s
                    tck = splrep(qty_raster, T_raster_c, k = k_s)
                    x = 'x_qty'
                    #derv = splev(qty_raster, tck, der = 1)/np.mean(qty_raster)
                    derv = np.gradient(T_raster_c)/np.gradient(qty_raster)*np.mean(qty_raster)/np.mean(T_raster_c)
                    print derv
                    if np.any(np.abs(derv) > der_cutoff):
                        x = 'x_T'
                except ValueError:
                    x = 'x_T'
                if x == 'x_qty':
                    val_init = qty_raster[np.argmin((T_raster_c - Tc)**2.0)]
                    print 'initial val:', val_init
                    val_at_Tc = amoeba([val_init], [val_init/40.0], val_search_at_Tc, data = [tck, Tc], xtolerance=1e-6, ftolerance=1e-7)[0][0]
                    print 'val at Tc:', val_at_Tc
                if x == 'x_T':
                    if T_raster_c[-1] < T_raster_c[0]:
                        T_raster_c = T_raster_c[::-1]
                        qty_raster = qty_raster[::-1]
                    tck = splrep(T_raster_c, qty_raster, k = k_s)
                    val_at_Tc = splev(Tc, tck)

                sn_Phiphi_Tc[ph][qty][i] = val_at_Tc
                if qty == 'Phi' or qty == 'phi':
                    figure(j)
                    plot(T_raster_c, qty_raster, label = str(mu)+' '+ph+' '+qty+' '+x)
                    if x == 'x_qty':
                        qty_r2 = np.linspace(qty_raster[0], qty_raster[-1], 400)
                        plot(splev(qty_r2, tck), qty_r2, ls = 'dashed')
                    elif x == 'x_T':
                        plot(Tr2s[ph], splev(Tr2s[ph], tck), ls = 'dashed')
                    scatter(Tc, val_at_Tc)
                    legend(frameon = False, loc = 'best')

                j += 1
                #sn_Phiphi_tcks[ph].update({qty:splrep(sn_Phiphi_rasters[qty][cut_inds[ph][0]:cut_inds[ph][1]], T_raster[cut_inds[ph][0]:cut_inds[ph][1]], k = k_s)})

    return sn_Phiphi_Tc

nzi = np.nonzero(Tc_muT)
Tc_muT = np.vstack((Tc_muT[nzi][:len(nzi[0])/2.0], Tc_muT[nzi][len(nzi[0])/2.0:]))
muT_ltp_spinodal = np.vstack((muT_ltp_spinodal[nzi][:len(nzi[0])/2.0], muT_ltp_spinodal[nzi][len(nzi[0])/2.0:]))
muT_htp_spinodal = np.vstack((muT_htp_spinodal[nzi][:len(nzi[0])/2.0], muT_htp_spinodal[nzi][len(nzi[0])/2.0:]))
sT3_ltp_spinodal = sT3_ltp_spinodal[np.nonzero(sT3_ltp_spinodal)]
sT3_htp_spinodal = sT3_htp_spinodal[np.nonzero(sT3_htp_spinodal)]
# for i in range(0, 2):
#     Phiphi_ltp_spinodal[i, :] = Phiphi_ltp_spinodal[i, :][np.nonzero(Phiphi_ltp_spinodal[i, :])]
#     Phiphi_htp_spinodal[i, :] = Phiphi_htp_spinodal[i, :][np.nonzero(Phiphi_htp_spinodal[i, :])]

Phiphi_ltp_spinodal = np.vstack((Phiphi_ltp_spinodal[nzi][:len(nzi[0])/2.0], Phiphi_ltp_spinodal[nzi][len(nzi[0])/2.0:]))
Phiphi_htp_spinodal = np.vstack((Phiphi_htp_spinodal[nzi][:len(nzi[0])/2.0], Phiphi_htp_spinodal[nzi][len(nzi[0])/2.0:]))

print 5*'#'
#print Phiphi_ltp_spinodal
#print Phiphi_htp_spinodal
#print buh

crvs = get_s_and_n_ltphtp_Tc(p_rasters_const_mu, phasetrans, Tc_muT)
#print crvs
#Phiphis = crvs[-1]

#show()

figure(1)
plot(Tc_muT[0, :], Tc_muT[1, :])
plot(muT_ltp_spinodal[0, :], muT_ltp_spinodal[1, :])
plot(muT_htp_spinodal[0, :], muT_htp_spinodal[1, :])
figure(20)
plot(Tc_muT[0, :], sT3_ltp_spinodal)
plot(Tc_muT[0, :], sT3_htp_spinodal)
for ph in ['ltp', 'htp', 'up']:
    plot(Tc_muT[0, :], crvs[ph]['sT3'])
figure(21)
for ph in ['ltp', 'htp', 'up']:
    plot(Tc_muT[0, :], crvs[ph]['nT3'])
#plot(Tc_muT[0, :], nT3_ltp_spinodal)
#plot(Tc_muT[0, :], nT3_htp_spinodal)

#from fmg_plotter_Tmu_wisen6 import fig19

figure(19)
plot(Phiphi_ltp_spinodal[0,:], np.log(Phiphi_ltp_spinodal[1,:]))
plot(Phiphi_htp_spinodal[0,:], np.log(Phiphi_htp_spinodal[1,:]))
for ph in ['ltp', 'htp', 'up']:
    plot(crvs[ph]['Phi'], np.log(crvs[ph]['phi']))
    #plot(Phiphis[i][0, :], np.log(Phiphis[i][1, :]))

p_on_levels = {'T':T_levels, 'mu':mu_levels, 'p_along_T':p_rasters_const_T, 'p_along_mu':p_rasters_const_mu}

print phasetrans
phase_contour_and_spinodals = {'spinodals':{'mu_T':{'ltp':muT_ltp_spinodal, 'htp':muT_htp_spinodal},
                                            'sT3':{'ltp':sT3_ltp_spinodal, 'htp':sT3_htp_spinodal},
                                            'phi':{'ltp':Phiphi_ltp_spinodal[1,:], 'htp':Phiphi_htp_spinodal[1,:]},
                                            'Phi':{'ltp':Phiphi_ltp_spinodal[0,:], 'htp':Phiphi_htp_spinodal[0,:]}},
                               'phase_contour':{'Tc_mu_T':Tc_muT, 'sn_Phiphi':crvs},
                               'p_Tc_data':phasetrans}

fname = model_type+'/'+ftype+'/p_on_levels_'+model_type+'.p'
pickle.dump(p_on_levels, open(fname, "wb"))
file.close(open(fname))
fname = model_type+'/'+ftype+'/phase_contour_and_spinodals_'+model_type+'.p'
pickle.dump(phase_contour_and_spinodals, open(fname, "wb"))
file.close(open(fname))

show()

###
print '\ndone: fmg_p_calc4.py'