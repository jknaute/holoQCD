import numpy as np
from scipy.interpolate import splrep, splev
import pickle
from pylab import figure, plot, show, legend, scatter

model_type = 'VRY_2'

fname = 'phase_contour_and_spinodals_'+model_type+'.p'
phase_contour_and_spinodals = pickle.load(open(fname, "rb"))
file.close(open(fname))

Tc_muT = phase_contour_and_spinodals['phase_contour']['Tc_mu_T']
sn_Phiphi_Tc = phase_contour_and_spinodals['phase_contour']['sn_Phiphi']
spinodals_muT = phase_contour_and_spinodals['spinodals']['mu_T']

phi_ltphtp = phase_contour_and_spinodals['spinodals']['phi']
Phi_ltphtp = phase_contour_and_spinodals['spinodals']['Phi']

Phiphi_curves_td = {'spinds':{'ltp':[Phi_ltphtp['ltp'], phi_ltphtp['ltp']], 'htp':[Phi_ltphtp['htp'], phi_ltphtp[
    'htp']]},
 'T_c':{'ltp':[sn_Phiphi_Tc['ltp']['Phi'], sn_Phiphi_Tc['ltp']['phi']],
        'htp':[sn_Phiphi_Tc['htp']['Phi'], sn_Phiphi_Tc['htp']['phi']],
        'up':[sn_Phiphi_Tc['up']['Phi'], sn_Phiphi_Tc['up']['phi']]}}

#print Tc_muT[0, :]

Tc_muT_tck = splrep(Tc_muT[0, :], Tc_muT[1, :],k=1)
ltp_spinodal_muT_tck = splrep(spinodals_muT['ltp'][0, :], spinodals_muT['ltp'][1, :],k=1)
htp_spinodal_muT_tck = splrep(spinodals_muT['htp'][0, :], spinodals_muT['htp'][1, :],k=1)

def CEP_from_intersection(Tc_muT_tck, ltp_spinodal_muT_tck, htp_spinodal_muT_tck, mu_offset, mu_minmax):
    mu_raster = np.linspace(mu_minmax[0] - mu_offset, mu_minmax[0], 2000)
    tcks = [Tc_muT_tck, ltp_spinodal_muT_tck, htp_spinodal_muT_tck]
    for i in range(0, len(tcks)):
        for j in range(i + 1, len(tcks)):
            CEP_loc_ind = np.argmin((splev(mu_raster, tcks[i]) - splev(mu_raster, tcks[j]))**2.0)
            T_cep_ltp = splev(mu_raster[CEP_loc_ind], ltp_spinodal_muT_tck)
            T_cep_htp = splev(mu_raster[CEP_loc_ind], htp_spinodal_muT_tck)
            T_cep = splev(mu_raster[CEP_loc_ind], Tc_muT_tck)
            print '\nT_CEP: ', i, j, mu_raster[CEP_loc_ind], T_cep_ltp, T_cep_htp, T_cep

    return

CEP_from_intersection(Tc_muT_tck, ltp_spinodal_muT_tck, htp_spinodal_muT_tck, 40.0, [Tc_muT[0, 0], Tc_muT[0, -1]])

figure(1)
plot(spinodals_muT['ltp'][0, :], spinodals_muT['ltp'][1, :], lw = 2, color = 'magenta')
plot(spinodals_muT['htp'][0, :], spinodals_muT['htp'][1, :], lw = 2, color = 'green')
plot(Tc_muT[0, :], Tc_muT[1, :], color = 'grey', lw = 2)

mu_raster2 = np.linspace(Tc_muT[0, 0] - 40, Tc_muT[0, -1], 2000)
plot(mu_raster2, splev(mu_raster2, ltp_spinodal_muT_tck), color = 'magenta', ls = 'dashed')
plot(mu_raster2, splev(mu_raster2, htp_spinodal_muT_tck), color = 'green', ls = 'dashed')
plot(mu_raster2, splev(mu_raster2, Tc_muT_tck), color = 'grey', ls = 'dashed')

figure(19)
# plot(Phi_ltphtp['ltp'], np.log(phi_ltphtp['ltp']))
# plot(Phi_ltphtp['htp'], np.log(phi_ltphtp['htp']))
# for ph in ['ltp', 'htp', 'up']:
#     plot(sn_Phiphi_Tc[ph]['Phi'], np.log(sn_Phiphi_Tc[ph]['phi']))

phases = ['ltp', 'htp', 'up']
for key in ['spinds', 'T_c']:
    phass = phases
    if key == 'spinds': phass = phases[:-1]
    for ph in phass:
        #print key, ph
        plot(Phiphi_curves_td[key][ph][0], Phiphi_curves_td[key][ph][1])
        if key == 'T_c' and ph == 'ltp':
            print 'buh'
        else:
            #print key, ph
            #print Phiphi_curves_td[key][ph][0]
            Phir_raster2 = np.linspace(Phiphi_curves_td[key][ph][0][0], Phiphi_curves_td[key][ph][0][-1], 20)
            if Phir_raster2[0] > Phir_raster2[-1]:
                Phir_raster2 = Phir_raster2[::-1]
            i_del = 0
            for i in range(0, len(Phiphi_curves_td[key][ph][0]) - 1):
                if Phiphi_curves_td[key][ph][0][i + 1] < Phiphi_curves_td[key][ph][0][i]:
                    i_del = i + 1
            #print 'deleting index', i_del, 'values', Phiphi_curves_td[key][ph][0][i_del], Phiphi_curves_td[key][ph][1][i_del]
            Phiphi_curves_td[key][ph][0] = np.delete(Phiphi_curves_td[key][ph][0], i_del)
            Phiphi_curves_td[key][ph][1] = np.delete(Phiphi_curves_td[key][ph][1], i_del)

            #print '\nx-array for splrep: ', Phiphi_curves_td[key][ph][0]
            Phiphi_curves_td[key][ph][0], Phiphi_curves_td[key][ph][1], Phir_raster2 = zip(*sorted(zip(Phiphi_curves_td[key][ph][0], Phiphi_curves_td[key][ph][1], Phir_raster2)))
            Phiphi_tck = splrep(Phiphi_curves_td[key][ph][0], Phiphi_curves_td[key][ph][1],s=0.1)

            plot(Phir_raster2, splev(Phir_raster2, Phiphi_tck), ls = 'dashed')

show()