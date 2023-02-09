import numpy as np
import numpy.ma as ma
from scipy.interpolate import splrep, splev
import pickle

from args_and_lambds import args_dic, lambdas_dic
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from pylab import figure, show, legend, plot, scatter

from TD_calc_pow import TD_calc_pointwise

model_type = 'VRY_2'
ftype = args_dic['ftype'][model_type]
V_args = args_dic['V'][model_type]
f_args = args_dic['f'][model_type]
lambdas = lambdas_dic[model_type]

if model_type == 'G':
    fname = 'phiPhi_range_htp_15.p'
    phiPhi_range_htp = pickle.load(open(fname, "rb"))
    file.close(open(fname))

    fname = 'phiPhi_range_ltp_15.p'
    phiPhi_range_ltp = pickle.load(open(fname, "rb"))
    file.close(open(fname))
elif model_type == 'VRY_2':
    fname = model_type+'/'+ftype+'/phiPhi_range_htp_VRY_2_2.p'
    phiPhi_range_htp = pickle.load(open(fname, "rb"))
    file.close(open(fname))

    fname = model_type+'/'+ftype+'/phiPhi_range_ltp_VRY_2_2.p'
    phiPhi_range_ltp = pickle.load(open(fname, "rb"))
    file.close(open(fname))

Phi1_pts = 20

print phiPhi_range_ltp
print phiPhi_range_htp
#print buh
phiPhi_ranges = {'ltp':phiPhi_range_ltp, 'htp':phiPhi_range_htp}
phi0_raster = {'ltp':phiPhi_range_ltp[0, :] , 'htp':phiPhi_range_htp[0, :]}
phiPhi_grid = {'ltp':np.zeros((len(phi0_raster['ltp']), Phi1_pts, 2)), 'htp':np.zeros((len(phi0_raster['htp']), Phi1_pts, 2))}

r_mids = {'ltp':np.zeros((len(phi0_raster['ltp']), Phi1_pts)), 'htp':np.zeros((len(phi0_raster['htp']), Phi1_pts))}
TD_grid = {'ltp':np.zeros((len(phi0_raster['ltp']), Phi1_pts, 4)), 'htp':np.zeros((len(phi0_raster['htp']), Phi1_pts, 4))}
print 'phi0_raster =', phi0_raster

ps = ['ltp', 'htp']

for ph in ps:
    r_mid = 18.0
    for i in range(0, len(phi0_raster[ph])):
        if ph == 'htp':
            if model_type == 'G':
                if i <= 8:
                    Phi1r_raster = np.linspace(0, (phiPhi_ranges[ph][2, i] - phiPhi_ranges[ph][1, i])**2.0, Phi1_pts)**(
                        1.0/2.0)
                    Phi1r_raster = phiPhi_ranges[ph][1, i] + Phi1r_raster
                else:
                    Phi1r_raster = np.linspace(phiPhi_ranges[ph][1, i], phiPhi_ranges[ph][2, i], Phi1_pts)
            elif model_type == 'VRY_2':
                Phi1r_raster = np.linspace(phiPhi_ranges[ph][1, i], phiPhi_ranges[ph][2, i], Phi1_pts)
        elif ph == 'ltp':
            Phi1r_raster = np.linspace(0, (phiPhi_ranges[ph][2, i] - phiPhi_ranges[ph][1, i])**2.0, Phi1_pts)**(1.0/2.0)
            Phi1r_raster = phiPhi_ranges[ph][1, i] + Phi1r_raster
            print phiPhi_ranges[ph][1, i], phiPhi_ranges[ph][2, i], Phi1r_raster
            #Phi1r_raster = np.linspace(phiPhi_ranges[ph][1, i]**4.0, phiPhi_ranges[ph][2, i]**4.0, Phi1_pts)**(1.0/4.0)
        print 'Phi1r_raster =', Phi1r_raster
        for j in range(0, len(Phi1r_raster)):
            phi_0 = phi0_raster[ph][i]
            Phi_1r = Phi1r_raster[j]
            print 'phase =', ph, 'phi_0 =', phi_0, 'Phi_1r =', Phi_1r
            phiPhi_grid[ph][i, j, :] = np.array([phi_0, Phi_1r])
            TD_pt = TD_calc_pointwise(phi_0, Phi_1r, V_args, f_args, r_mid)

            TD_grid[ph][i, j, :] = TD_pt[0]*lambdas
            r_mid = TD_pt[2]

            print 'TD vals:', TD_grid[ph][i, j, :]

fname = model_type+'/'+ftype+'/phiPhi_grid_ltp_htp_VRY_2_2.p'
pickle.dump(phiPhi_grid, open(fname, "wb"))
file.close(open(fname))

fname = model_type+'/'+ftype+'/TD_grid_ltp_htp_VRY_2_2.p'
pickle.dump(TD_grid, open(fname, "wb"))
file.close(open(fname))


print '\ndone: TD_calc_htp_ltp.py'
