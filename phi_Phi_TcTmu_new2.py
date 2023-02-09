import numpy as np
import numpy.ma as ma
from scipy.interpolate import splrep, splev
import pickle

from args_and_lambds import args_dic, lambdas_dic
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from pylab import figure, show, legend, plot, scatter

model_type = 'VRY_2'
ftype = args_dic['ftype'][model_type]
V_args = args_dic['V'][model_type]
f_args = args_dic['f'][model_type]
lambdas = lambdas_dic[model_type]

if model_type == 'G':
    fname = 'TD_gr_G_wmu0.p'
if model_type == 'no':
    fname = 'TD_gr_no.p'
if model_type == 'VRY_2':
    fname = model_type+'/'+ftype+'/TD_gr_VRY_2_wmu0.p'

print fname
TD_gr = pickle.load(open(fname, "rb"))
file.close(open(fname))

TD_grid = TD_gr[0]
TD_grid = TD_scale(TD_grid, lambdas)[0]
phiPhi_grid = TD_gr[1]
TD_full = TD_gr[2]

phi0_pts = TD_full['phi0_pts']
Phi1_pts = TD_full['Phi1_pts']

fname = model_type+'/'+ftype+'/phase_contour_and_spinodals_'+model_type+'.p'
phase_contour_and_spinodals = pickle.load(open(fname, "rb"))
file.close(open(fname))

print 'phase_contour: ', phase_contour_and_spinodals['phase_contour'].keys()
print 'sn_Phiphi: ', phase_contour_and_spinodals['phase_contour']['sn_Phiphi']
print 4*'#'
Tc_muT = phase_contour_and_spinodals['phase_contour']['Tc_mu_T']
sn_Pp_ltp_htp = phase_contour_and_spinodals['phase_contour']['sn_Phiphi']
spinodals_muT = phase_contour_and_spinodals['spinodals']['mu_T']

p_Tc_data = phase_contour_and_spinodals['p_Tc_data']
print p_Tc_data.keys()
#print bh

Tc_muT_tck = splrep(Tc_muT[0, :], Tc_muT[1, :])

phiPhi_ltp = np.vstack((sn_Pp_ltp_htp['ltp']['phi'], sn_Pp_ltp_htp['ltp']['Phi']))
phiPhi_htp = np.vstack((sn_Pp_ltp_htp['htp']['phi'][::-1], sn_Pp_ltp_htp['htp']['Phi'][::-1]))
print phiPhi_ltp
print phiPhi_htp

phi_mid = (phiPhi_ltp[0, 0] + phiPhi_htp[0, -1])/2.0
Phi_mid = (phiPhi_ltp[1, 0] + phiPhi_htp[1, -1])/2.0

print phi_mid, Phi_mid

a, b = np.hstack((phi_mid, phiPhi_ltp[0, :])), np.hstack((Phi_mid, phiPhi_ltp[1, :]))
a, b = zip(*sorted(zip(a, b)))
Phiphi_ltp_tck = splrep(a, b)
Phiphi_htp_tck = splrep(np.hstack((phiPhi_htp[0, :], phi_mid)), np.hstack((phiPhi_htp[1, :], Phi_mid)))

Phiphi_ltp_tck_k1 = splrep(a, b, k = 1)
Phiphi_htp_tck_k1 = splrep(np.hstack((phiPhi_htp[0, :], phi_mid)), np.hstack((phiPhi_htp[1, :], Phi_mid)), k = 1)

num_phi_newpts = 28
phi_ltp_max = phiPhi_ltp[0, -3]
phi_htp_min = phiPhi_htp[0, 2]
phi_ltp_min = phiPhi_ltp[0, 0]
phi_htp_max = phiPhi_htp[0, -1]
#phi_ltp_min = phi_mid
#phi_htp_max = phi_mid

phi_new_ltp = np.linspace(phi_ltp_min, phi_ltp_max, num_phi_newpts)
phi_new_htp = np.linspace(phi_htp_min, phi_htp_max, num_phi_newpts)

Phi1r_new_ltp_max = 1.02*splev(phi_new_ltp, Phiphi_ltp_tck)
Phi1r_new_htp_max = 1.02*splev(phi_new_htp, Phiphi_htp_tck)

extend_phi_range = 0
if extend_phi_range:
    phi_ltp_max2 = phi_ltp_max*1.02
    phi_htp_min2 = phi_htp_min*0.96
    num_phi_newpts += 4
    phi_new_ltp_2 = np.linspace(phi_ltp_max, phi_ltp_max2, 5)[1:]
    phi_new_htp_2 = np.linspace(phi_htp_min2, phi_htp_min, 5)[:-1]
    Phi1r_new_ltp_max_2 = 1.02*splev(phi_new_ltp_2, Phiphi_ltp_tck_k1)
    Phi1r_new_htp_max_2 = 1.02*splev(phi_new_htp_2, Phiphi_htp_tck_k1)

    phi_new_ltp = np.hstack((phi_new_ltp, phi_new_ltp_2))
    phi_new_htp = np.hstack((phi_new_htp_2, phi_new_htp))
    Phi1r_new_ltp_max = np.hstack((Phi1r_new_ltp_max, Phi1r_new_ltp_max_2))
    Phi1r_new_htp_max = np.hstack((Phi1r_new_htp_max_2, Phi1r_new_htp_max))

print phi_new_ltp
print Phi1r_new_ltp_max
print phi_new_htp
print Phi1r_new_htp_max

Phi_ltphtp_stretcher = np.linspace(0.8, 0.9, num_phi_newpts)

Phi1r_new_ltp_min = Phi_ltphtp_stretcher[::-1]*Phi1r_new_ltp_max
Phi1r_new_htp_min = Phi_ltphtp_stretcher*Phi1r_new_htp_max

ltp_newpts = np.vstack((phi_new_ltp, Phi1r_new_ltp_min, Phi1r_new_ltp_max))
htp_newpts = np.vstack((phi_new_htp, Phi1r_new_htp_min, Phi1r_new_htp_max))
#htp_newpts = phi0Phi1r_new_htp

print ltp_newpts
print htp_newpts

figure(3)
for i in range(0, len(ltp_newpts[0, :])):
    Phi1min = ltp_newpts[1, i]
    Phi1max = ltp_newpts[2, i]
    phi0 = ltp_newpts[0, i]
    plot(np.array([Phi1min, Phi1max]), np.array([phi0, phi0]), color = 'green')
for i in range(0, len(ltp_newpts[0, :])):
    Phi1min = htp_newpts[1, i]
    Phi1max = htp_newpts[2, i]
    phi0 = htp_newpts[0, i]
    plot(np.array([Phi1min, Phi1max]), np.array([phi0, phi0]), color = 'blue')

for ph in ['ltp', 'htp', 'up']:
    plot(sn_Pp_ltp_htp[ph]['Phi'], sn_Pp_ltp_htp[ph]['phi'])

figure(4)
plot(Tc_muT[0, :], Tc_muT[1, :])
plot(spinodals_muT['ltp'][0, :], spinodals_muT['ltp'][1, :])
plot(spinodals_muT['htp'][0, :], spinodals_muT['htp'][1, :])

## (phi_0, Phi_1r)(p_c(T, u)) ltp
if not extend_phi_range:
    ltp_pc_onnewpts = np.vstack((phi_new_ltp, splev(phi_new_ltp, Phiphi_ltp_tck))) # <===
else:
    ltp_pc_onnewpts1 = np.vstack((phi_new_ltp[:-4], splev(phi_new_ltp[:-4], Phiphi_ltp_tck))) # <===
    ltp_pc_onnewpts2 = np.vstack((phi_new_ltp[-4:], splev(phi_new_ltp[-4:], Phiphi_ltp_tck_k1))) # <===
    ltp_pc_onnewpts = np.hstack((ltp_pc_onnewpts1, ltp_pc_onnewpts2))

## (phi_0, Phi_1r)(p_c(T, u)) htp
if not extend_phi_range:
    htp_pc_onnewpts = np.vstack((phi_new_htp, splev(phi_new_htp, Phiphi_htp_tck))) # <===
else:
    htp_pc_onnewpts1 = np.vstack((phi_new_htp[:4], splev(phi_new_htp[:4], Phiphi_htp_tck_k1))) # <===
    htp_pc_onnewpts2 = np.vstack((phi_new_htp[4:], splev(phi_new_htp[4:], Phiphi_htp_tck))) # <===
    htp_pc_onnewpts = np.hstack((htp_pc_onnewpts1, htp_pc_onnewpts2))

figure(3)
plot(ltp_pc_onnewpts[1, :], ltp_pc_onnewpts[0, :])
plot(htp_pc_onnewpts[1, :], htp_pc_onnewpts[0, :])

###
logpc_pts = np.zeros(len(Tc_muT[0, :]))
k = 0
for mu_val in Tc_muT[0, :]:
    print mu_val, phase_contour_and_spinodals['p_Tc_data'][mu_val]['p_scaled']
    logpc_pts[k] = np.log(phase_contour_and_spinodals['p_Tc_data'][mu_val]['p'])
    #logpc_pts[k] = phase_contour_and_spinodals['p_Tc_data'][mu_val]['p_scaled']
    k+=1

figure(31)
plot(sn_Pp_ltp_htp['ltp']['phi'], logpc_pts)
show()
sn_Pp_ltp_htp['ltp']['phi'], logpc_pts = zip(*sorted(zip(sn_Pp_ltp_htp['ltp']['phi'], logpc_pts)))
logpc_ltp_tck = splrep(sn_Pp_ltp_htp['ltp']['phi'], logpc_pts)
logpc_htp_tck = splrep(sn_Pp_ltp_htp['htp']['phi'][::-1], logpc_pts[::-1])
logpc_ltp_tck_k1 = splrep(sn_Pp_ltp_htp['ltp']['phi'], logpc_pts, k = 1)
logpc_htp_tck_k1 = splrep(sn_Pp_ltp_htp['htp']['phi'][::-1], logpc_pts[::-1], k = 1)

Tc_phi_ltp_tck = splrep(sn_Pp_ltp_htp['ltp']['phi'], Tc_muT[1, :])
Tc_phi_htp_tck = splrep(sn_Pp_ltp_htp['htp']['phi'][::-1], Tc_muT[1, :][::-1])
Tc_phi_ltp_tck_k1 = splrep(sn_Pp_ltp_htp['ltp']['phi'], Tc_muT[1, :], k = 1)
Tc_phi_htp_tck_k1 = splrep(sn_Pp_ltp_htp['htp']['phi'][::-1], Tc_muT[1, :][::-1], k = 1)

if not extend_phi_range:
    ltp_logpc_newpts = splev(phi_new_ltp, logpc_ltp_tck)
    htp_logpc_newpts = splev(phi_new_htp, logpc_htp_tck)
    ltp_Tc_newpts = splev(phi_new_ltp, Tc_phi_ltp_tck)
    htp_Tc_newpts = splev(phi_new_htp, Tc_phi_htp_tck)
else:
    ltp_logpc_newpts = np.hstack((splev(phi_new_ltp[:-4], logpc_ltp_tck), splev(phi_new_ltp[-4:], logpc_ltp_tck_k1)))
    htp_logpc_newpts = np.hstack((splev(phi_new_htp[:4], logpc_htp_tck_k1), splev(phi_new_htp[4:], logpc_htp_tck)))

    ltp_Tc_newpts1 = splev(phi_new_ltp[:-4], Tc_phi_ltp_tck)
    ltp_Tc_newpts2 = splev(phi_new_ltp[-4:], Tc_phi_ltp_tck_k1)

    htp_Tc_newpts1 = splev(phi_new_htp[:4], Tc_phi_htp_tck_k1)
    htp_Tc_newpts2 = splev(phi_new_htp[4:], Tc_phi_htp_tck)

    ltp_Tc_newpts = np.hstack((ltp_Tc_newpts1, ltp_Tc_newpts2))
    htp_Tc_newpts = np.hstack((htp_Tc_newpts1, htp_Tc_newpts2))


figure(5)
plot(sn_Pp_ltp_htp['ltp']['phi'], np.log(logpc_pts*Tc_muT[1, :]**4.0), color = 'green')
plot(phi_new_ltp, np.log(ltp_logpc_newpts*ltp_Tc_newpts**4.0), color = 'green', ls = 'dashed')
plot(sn_Pp_ltp_htp['htp']['phi'][::-1], np.log(logpc_pts[::-1]*Tc_muT[1, :][::-1]**4.0), color = 'blue')
plot(phi_new_htp, np.log(htp_logpc_newpts*htp_Tc_newpts**4.0), color = 'blue', ls = 'dashed')

figure(6)
plot(sn_Pp_ltp_htp['ltp']['phi'], logpc_pts, color = 'green')
plot(phi_new_ltp, ltp_logpc_newpts, color = 'green', ls = 'dashed')
plot(sn_Pp_ltp_htp['htp']['phi'][::-1], logpc_pts[::-1], color = 'blue')
plot(phi_new_htp, htp_logpc_newpts, color = 'blue', ls = 'dashed')

ltp_newpts_pcline = np.vstack((ltp_pc_onnewpts, ltp_logpc_newpts))
htp_newpts_pcline = np.vstack((htp_pc_onnewpts, htp_logpc_newpts))

print 10*'#'
print ltp_newpts_pcline
print htp_newpts_pcline

pclines = {'ltp':ltp_newpts_pcline, 'htp':htp_newpts_pcline}

fname = model_type+'/'+ftype+'/phiPhi_range_htp_VRY_2_2.p'
pickle.dump(htp_newpts, open(fname, "wb"))
file.close(open(fname))
fname = model_type+'/'+ftype+'/phiPhi_range_ltp_VRY_2_2.p'
pickle.dump(ltp_newpts, open(fname, "wb"))
file.close(open(fname))

fname = model_type+'/'+ftype+'/pc_lines_VRY_2_2.p'
pickle.dump(pclines, open(fname, "wb"))
file.close(open(fname))


show()