import numpy as np
import numpy.ma as ma
from scipy.interpolate import splrep, splev
import pickle
from nice_tcks import nice_ticks

from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp, subplots, gcf, tight_layout, subplots_adjust, grid, title, clabel
from matplotlib.colors import Normalize
from fmg_TDprocess import TD_scale
from masker_for_phase_diagram4 import get_stable_phases_for_PD
from args_and_lambds import args_dic, lambdas_dic

def get_contour_verts(cn, xy_rev, path_rev):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                if xy_rev:
                    xy.append(vv[0][::-1])
                else:
                    xy.append(vv[0])
            if path_rev:
                paths.append(np.vstack(xy)[::-1])
            else:
                paths.append(np.vstack(xy))
        contours.append(paths)

    return contours

## layout for PDs:
rc('font', size = 30) #fontsize of axis labels (numbers)
rc('axes', labelsize = 38, lw = 2) #fontsize of axis labels (symbols)
rc('xtick.major', pad = 2)
rc('ytick.major', pad = 7)

MS = 5      # marker size
MEW = 1     # marker edge width

## layout for small diagrams:
#linew = 3
#rc('font', size = 20) #fontsize of axis labels (numbers)
#rc('axes', labelsize = 22, lw = linew) #fontsize of axis labels (symbols)
#rc('lines', mew = 2, lw = linew, markeredgewidth = 2)
#rc('xtick.major', pad = 7)
#rc('ytick.major', pad = 7)

lw_for_pd = 10
CEP_dotsize = 35**2
clabel_fs = 18      # fontsize
set_labels_manually = 0      ##  <====================================   CHOOSE
save_figs = 1
plot_bckgrnd = 1    # draw ltp phase separately
plot_small_excerpt = 0  # for comparison w/ lattice at small mu (s/n) [choose one excerpt or nothing]
plot_large_excerpt = 1  #     -- "" --      hee result (s/T^3)

#+++++++++++++++
model_type = 'VRY_2'
ftype = args_dic['ftype'][model_type]
lambdas = lambdas_dic['VRY_2']
#+++++++++++++++


fname = model_type+'/'+ftype+'/TD_grid_masked_VRY_2.p'
TD_grid_m = pickle.load(open(fname, "rb"))
file.close(open(fname))

fname = model_type+'/'+ftype+'/p_grid_masked_VRY_2.p'
p_grid_m = pickle.load(open(fname, "rb"))
file.close(open(fname))

fname = model_type+'/'+ftype+'/TD_grid_masked_VRY_2_ltp.p'
TD_grid_m_ltp = pickle.load(open(fname, "rb"))
file.close(open(fname))

fname = model_type+'/'+ftype+'/p_grid_masked_VRY_2_ltp.p'
p_grid_m_ltp = pickle.load(open(fname, "rb"))
file.close(open(fname))



fname = model_type+'/'+ftype+'/TD_grid_ltp_htp_VRY_2_2.p'
TD_grid_ltp_htp = pickle.load(open(fname, "rb"))
file.close(open(fname))

fname = model_type+'/'+ftype+'/phiPhi_grid_ltp_htp_VRY_2_2.p'
phiPhi_grid_ltp_htp = pickle.load(open(fname, "rb"))
file.close(open(fname))

fname = model_type+'/'+ftype+'/p_grid_ltp_htp_VRY_2_2.p'
p_grid_ltp_htp = pickle.load(open(fname, "rb"))
file.close(open(fname))



masked_ltp = get_stable_phases_for_PD(TD_grid_ltp_htp['ltp'], phiPhi_grid_ltp_htp['ltp'], p_grid_ltp_htp['ltp'], 'all')
masked_htp = get_stable_phases_for_PD(TD_grid_ltp_htp['htp'], phiPhi_grid_ltp_htp['htp'], p_grid_ltp_htp['htp'], 'all')

TD_grid_ltp_m = masked_ltp[0]
TD_grid_htp_m = masked_htp[0]
p_grid_ltp_m = masked_ltp[2]
p_grid_htp_m = masked_htp[2]

TD_grids_m = [TD_grid_m, TD_grid_ltp_m, TD_grid_htp_m]
p_grids_m = [p_grid_m, p_grid_ltp_m, p_grid_htp_m]


fname = model_type+'/'+ftype+'/TD_gr_VRY_2_wmu0.p'
TD_gr = pickle.load(open(fname, "rb"))
file.close(open(fname))
TD_grid = TD_gr[0]
TD_grid = TD_scale(TD_grid, lambdas)[0]
phiPhi_grid = TD_gr[1]
phi0_pts = TD_gr[2]['phi0_pts']
Phi1_pts = TD_gr[2]['Phi1_pts']

fname = model_type+'/'+ftype+'/phase_contour_and_spinodals_VRY_2.p'
phase_contour_and_spinodals = pickle.load(open(fname, "rb"))
file.close(open(fname))

Tc_muT = phase_contour_and_spinodals['phase_contour']['Tc_mu_T']
#sn_ltp_htp = phase_contour_and_spinodals['phase_contour']['s_n']
spinodals_muT =  phase_contour_and_spinodals['spinodals']['mu_T']
s_n_Tc_htpltp = phase_contour_and_spinodals['phase_contour']['sn_Phiphi']
p_Tc_data = phase_contour_and_spinodals['p_Tc_data']


ltp_spinodal_muT_tck = splrep(spinodals_muT['ltp'][0, :], spinodals_muT['ltp'][1, :])
htp_spinodal_muT_tck = splrep(spinodals_muT['htp'][0, :], spinodals_muT['htp'][1, :])

Tc_muT_tck = splrep(Tc_muT[0, :], Tc_muT[1, :])

#print TD_grid_m[-4:, :, :]

T_raster_mu0 = TD_grid_m[:, 0, 0][::-1]
print T_raster_mu0
#print buh
print 'T_mu0 =', T_raster_mu0
sT3_T_mu0 = TD_grid_m[:, 0, 1][::-1]/T_raster_mu0**3.0
nT3_T_musmall = TD_grid_m[:, 1, 3][::-1]/T_raster_mu0**3.0
sT3_globalmax = np.amax(TD_grid_m[:, :, 1]/TD_grid_m[:, :, 0]**3.0)
nT3_globalmax = np.amax(TD_grid_m[:, :, 3]/TD_grid_m[:, :, 0]**3.0)
print 'sT3_globalmax = %2.8f, nT3_globalmax = %2.8f' %(sT3_globalmax, nT3_globalmax)
Tm0_nz = np.nonzero(T_raster_mu0)
print 'nT3 musmall', nT3_T_musmall

T_raster_mu0 = T_raster_mu0[Tm0_nz]
sT3_T_mu0 = sT3_T_mu0[Tm0_nz]
nT3_T_mus = nT3_T_musmall[Tm0_nz]

print T_raster_mu0, len(T_raster_mu0)
print nT3_T_mus.argmax(), nT3_T_mus.max()

sT3_mu0_tck = splrep(T_raster_mu0, sT3_T_mu0)
nT3_mus_tck = splrep(T_raster_mu0, nT3_T_mus)
T_range = [T_raster_mu0[0], T_raster_mu0[-1]]


### sT3 levels:
sT3_levels1 = splev(np.linspace(T_range[0], T_range[-1], 500), sT3_mu0_tck)
sT3_levels2 = np.linspace(splev(T_range[-1], sT3_mu0_tck)*1.001, 25.0, 80)
sT3_levels = np.hstack((sT3_levels1, sT3_levels2))

#sT3_levels_2_a = np.array([0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0])
sT3_levels_2_a = np.array([1.8, 2.0, 2.5, 3.0, 3.5]) # np.array([1.5, 1.7, 2.0, 2.5, 3.0])
sT3_levels_2_b = np.arange(4.0, 20.5, 2.0)
sT3_levels_2_c = np.array([22.0])
sT3_levels_2 = np.hstack((sT3_levels_2_a, sT3_levels_2_b, sT3_levels_2_c))
#sT3_levels_2 = np.sort(np.insert(sT3_levels_2, [0, 1, 2, 3, 4, 5, 6], [10.0, 14.0, 18.0, 22.0, 5, 6, 7]))

sT3_colors_2 = [None]*len(sT3_levels_2)
for i in range(0, len(sT3_colors_2)):
    sT3_colors_2[i] = cm.Greens(sT3_levels_2[i]/np.amax(sT3_levels_2))


### sT3 levels excerpt:
sT3_levels1 = np.linspace(0, 5, 200)
sT3_levels2 = np.linspace(5, 30.0, 200)
sT3_levels_excerpt = np.hstack((sT3_levels1, sT3_levels2))

sT3_levels_2_a = np.array([1.0, 1.5, 1.7, 2.0, 2.5, 3.0])
sT3_levels_2_b = np.arange(4.0, 20.5, 2.0)
sT3_levels_2_c = np.array([22.0])
sT3_levels_2_excerpt = np.hstack((sT3_levels_2_a, sT3_levels_2_b, sT3_levels_2_c))

sT3_colors_2_excerpt = [None]*len(sT3_levels_2_excerpt)
for i in range(0, len(sT3_colors_2_excerpt)):
    sT3_colors_2_excerpt[i] = cm.Greens(sT3_levels_2_excerpt[i]/np.amax(sT3_levels_2_excerpt))


### nT3 levels:
nT3_levels1a = splev(np.linspace(T_range[0], T_raster_mu0[nT3_T_mus.argmax()], 200), nT3_mus_tck)
nT3_levels1b = np.linspace(nT3_T_mus.max(), 0.3, 200)
nT3_levels1 = np.hstack((nT3_levels1a, nT3_levels1b))
nT3_levels2 = np.linspace(0.3001, 6.0, 200)
nT3_levels = np.hstack((nT3_levels1, nT3_levels2))

nT3_levels_2_a = np.array([0.1, 0.15, 0.2, 0.3]) # np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
nT3_levels_2_b = np.arange(0.4, 1.0, 0.2)
nT3_levels_2_c = np.arange(1.0, 4.61, 0.4)
nT3_levels_2_d = [] #np.arange(6.0, 12.01, 1.0)
nT3_levels_2 = np.hstack((nT3_levels_2_a, nT3_levels_2_b, nT3_levels_2_c, nT3_levels_2_d))

nT3_colors_2 = [None]*len(nT3_levels_2)
for i in range(0, len(nT3_colors_2)):
    nT3_colors_2[i] = cm.Greens(nT3_levels_2[i]/np.amax(nT3_levels_2))


### s/n levels:
sn_levels1 = np.linspace(1.0, 15.0, 200)
sn_levels2 = np.linspace(15.1, 26, 100) # np.linspace(15.1, 100, 100)
sn_levels = np.hstack((sn_levels1, sn_levels2))

isen_levels = np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0]) # np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0])
isen_levels_forlog = np.array([4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 20.0]) # np.array([4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 20.0, 30.0, 50.0])

isen_colors = len(isen_levels)*[None]
for i in range(0, len(isen_colors)):
    isen_colors[i] = cm.Greens(isen_levels[i]/np.amax(isen_levels))
isen_colors_forlog = len(isen_levels_forlog)*[None]
for i in range(0, len(isen_colors_forlog)):
    isen_colors_forlog[i] = cm.Greens(isen_levels_forlog[i]/np.amax(isen_levels_forlog))

### s/n levels in excerpt for comparison with lattice data:
sn_levels1 = np.linspace(1.0, 15.0, 100)
sn_levels2 = np.linspace(15.1, 600.0, 300)
sn_levels_excerpt = np.hstack((sn_levels1, sn_levels2))

isen_levels_excerpt = np.array([30.0, 51.0, 70.0, 94.0, 144.0, 420.0])

isen_colors_excerpt = len(isen_levels_excerpt)*[None]
for i in range(0, len(isen_colors_excerpt)):
    isen_colors_excerpt[i] = cm.Greens(isen_levels_excerpt[i]/np.amax(isen_levels_excerpt))
isen_colors_excerpt2 = ['b', 'g', 'r', 'c', 'darkviolet', 'olive']

### p/T^4 levels:
pT4_levels1 = np.linspace(0, 1.5, 200)
pT4_levels2 = np.linspace(1.501, 50.0, 200)
pT4_levels = np.hstack((pT4_levels1, pT4_levels2))
#pT4_levels2 = np.linspace(0, 0.06, 200)
#pT4_levels = np.hstack((pT4_levels2))

pT4_levels_2a = np.array([0.6, 0.8, 1.0, 1.5, 3.0]) # np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 3.0])
pT4_levels_2b = np.arange(2.0, 8.1, 2.0)
pT4_levels_2c = np.arange(10.0, 30.1, 5.0)
pT4_levels_2  = np.hstack((pT4_levels_2a, pT4_levels_2b, pT4_levels_2c))

pT4_colors_2 = [None]*len(pT4_levels_2)
for i in range(0, len(pT4_colors_2)):
    pT4_colors_2[i] = cm.Greens(pT4_levels_2[i]/np.amax(pT4_levels_2))


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ATTENTION: Update CEP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if model_type == 'G':
    T_CEP = 143.0
    mu_CEP = 783.0
elif model_type == 'VRY_2':
    T_CEP = 111.5
    mu_CEP = 988.9
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


T_min = 0.5#*T_CEP
T_max = 1.6#*T_CEP
mu_min = 0.5#*mu_CEP
mu_max = 1.43#*mu_CEP

for i in range(0, phi0_pts):
    for j in range(0, Phi1_pts):
        TD_grid_m[i, j, 0] = TD_grid_m[i, j, 0]/T_CEP
        TD_grid_m[i, j, 2] = TD_grid_m[i, j, 2]/mu_CEP
        TD_grid_m_ltp[i, j, 0] = TD_grid_m_ltp[i, j, 0]/T_CEP
        TD_grid_m_ltp[i, j, 2] = TD_grid_m_ltp[i, j, 2]/mu_CEP

        if plot_small_excerpt:
            if TD_grid_m[i, j, 0]*T_CEP > 305.0 or TD_grid_m[i, j, 0] < T_min*0.86 or TD_grid_m[i, j, 2] > mu_max*1.15 or TD_grid_m[i, j, 2] < 0.0:
                TD_grid_m[i, j, :] = ma.masked
                p_grid_m[i, j] = ma.masked
                TD_grid_m_ltp[i, j, :] = ma.masked
                p_grid_m_ltp[i, j] = ma.masked
        elif plot_large_excerpt:
            if TD_grid_m[i, j, 0]*T_CEP > 210.0 or TD_grid_m[i, j, 0]*T_CEP < 40.0 or TD_grid_m[i, j, 2]*mu_CEP > 1600 or TD_grid_m[i, j, 2] < 0.0:
                TD_grid_m[i, j, :] = ma.masked
                p_grid_m[i, j] = ma.masked
                TD_grid_m_ltp[i, j, :] = ma.masked
                p_grid_m_ltp[i, j] = ma.masked
        else:
            if TD_grid_m[i, j, 0] > T_max*1.1 or TD_grid_m[i, j, 0] < T_min*0.86 or TD_grid_m[i, j, 2] > mu_max*1.15 or TD_grid_m[i, j, 2] < mu_min*0.86:
                TD_grid_m[i, j, :] = ma.masked
                p_grid_m[i, j] = ma.masked
                TD_grid_m_ltp[i, j, :] = ma.masked
                p_grid_m_ltp[i, j] = ma.masked

for TD_grid_pd in TD_grids_m[1:]:
    for i in range(0, len(TD_grid_pd[:, 0, 0])):
        for j in range(0, len(TD_grid_pd[0, :, 0])):
            TD_grid_pd[i, j, 0] = TD_grid_pd[i, j, 0]/T_CEP
            TD_grid_pd[i, j, 2] = TD_grid_pd[i, j, 2]/mu_CEP

fixed_ticks = 1
if fixed_ticks:
    sT3_ticks = list(np.arange(0, np.amax(sT3_levels)*1.001, np.amax(sT3_levels)/10.0))
    sT3_ticks_excerpt = list(np.arange(0, np.amax(sT3_levels_excerpt)*1.001, np.amax(sT3_levels_excerpt)/10.0))
    nT3_ticks = list(np.arange(0, np.amax(nT3_levels)*1.001, np.amax(nT3_levels)/10.0))
    sn_ticks = list(np.arange(0, np.amax(sn_levels)*1.001, np.amax(sn_levels)/10.0))
    sn_ticks_excerpt = list(np.arange(0, np.amax(sn_levels_excerpt)*1.001, np.amax(sn_levels_excerpt)/10.0))
    pT4_ticks = list(np.arange(0, np.amax(pT4_levels)*1.001, np.amax(pT4_levels)/10.0))
else:
    sT3_ticks = None
    sT3_ticks_excerpt = None
    nT3_ticks = None
    sn_ticks = None
    sn_ticks_excerpt = None
    pT4_ticks = None

print np.amax(TD_grid_m[:, :, 0])
ff = 1.7
print 'min/max sT3:', np.amin(TD_grid_m[:, :, 1]/TD_grid_m[:, :, 0]**3.0), np.amax(TD_grid_m[:, :, 1]/TD_grid_m[:, :, 0]**3.0)
#norm = Normalize(vmin = np.amin(TD_grid_m[:, :, 1]/TD_grid_m[:, :, 0]**3.0), vmax = 80.0)


##### s/T^3 #####
figure(1, figsize = (ff*8.0, ff*6.2))
if plot_bckgrnd:
    contourf(TD_grid_m_ltp[:, :, 2], TD_grid_m_ltp[:, :, 0], TD_grid_m_ltp[:, :, 1]/TD_grid_m_ltp[:, :, 0]**3.0/T_CEP**3.0, levels = sT3_levels, cmap = cm.jet, linewidths = 5, zorder=-4)
for k in range(0, 2):
    for TD_grid_pd in TD_grids_m[::-1]:
        contourf(TD_grid_pd[:, :, 2], TD_grid_pd[:, :, 0], TD_grid_pd[:, :, 1]/TD_grid_pd[:, :, 0]**3.0/T_CEP**3.0,
                 levels = sT3_levels, cmap = cm.jet, linewidths = 5, zorder=-3)

nice_ticks()
axis([mu_min, mu_max, T_min, T_max])
colorbar(spacing = 'proportional', ticks = sT3_ticks)
plot(Tc_muT[0, :]/mu_CEP, Tc_muT[1, :]/T_CEP, color = 'grey', lw = lw_for_pd, zorder=-1)

for TD_grid_pd in TD_grids_m[::-1]:
    sT3_lvls = contour(TD_grid_pd[:, :, 2], TD_grid_pd[:, :, 0], TD_grid_pd[:, :, 1]/TD_grid_pd[:, :, 0]**3.0/T_CEP**3.0,
                 levels = sT3_levels_2, colors = sT3_colors_2, linewidths = 3, zorder=-2)

# fmts = {}
# for sT3lv in sT3_levels_2[:5]:
#     fmts.update({sT3lv:str(sT3lv)})
# clabel(sT3_lvls, sT3_levels_2[:5], inline = 1, fontsize = 12, fmt = fmts)
clabel(sT3_lvls, sT3_levels_2, inline = 1, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually)
scatter(1.0, 1.0, c='white', s=CEP_dotsize, alpha=1, zorder=1, edgecolor='white')

#plot(spinodals_muT['ltp'][0, :], spinodals_muT['ltp'][1, :], lw = 2, color = 'magenta')
#plot(spinodals_muT['htp'][0, :], spinodals_muT['htp'][1, :], lw = 2, color = 'yellow')



#####   s/T^3  EXCERPT   #####
figure(11, figsize = (ff*8.0, ff*6.2))
if plot_bckgrnd:
    contourf(TD_grid_m_ltp[:, :, 2], TD_grid_m_ltp[:, :, 0], TD_grid_m_ltp[:, :, 1]/TD_grid_m_ltp[:, :, 0]**3.0/T_CEP**3.0,
             levels = sT3_levels_excerpt, cmap = cm.jet, linewidths = 5, zorder=-4)
for k in range(0, 2):
    for TD_grid_pd in TD_grids_m[::-1]:
        contourf(TD_grid_pd[:, :, 2], TD_grid_pd[:, :, 0], TD_grid_pd[:, :, 1]/TD_grid_pd[:, :, 0]**3.0/T_CEP**3.0,
                 levels = sT3_levels_excerpt, cmap = cm.jet, linewidths = 5, zorder=-3)

nice_ticks()
axis([0, 1450/mu_CEP, 50/T_CEP, 200/T_CEP])
colorbar(spacing = 'proportional', ticks = sT3_ticks_excerpt)
plot(Tc_muT[0, :]/mu_CEP, Tc_muT[1, :]/T_CEP, color = 'grey', lw = lw_for_pd, zorder=-1)

for TD_grid_pd in TD_grids_m[::-1]:
    sT3_lvls = contour(TD_grid_pd[:, :, 2], TD_grid_pd[:, :, 0], TD_grid_pd[:, :, 1]/TD_grid_pd[:, :, 0]**3.0/T_CEP**3.0,
                 levels = sT3_levels_2_excerpt, colors = sT3_colors_2_excerpt, linewidths = 3, zorder=-2)

clabel(sT3_lvls, sT3_levels_2_excerpt, inline = 1, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually)
scatter(1.0, 1.0, c='white', s=CEP_dotsize, alpha=1, zorder=1, edgecolor='white')



##### n/T^3 #####
figure(2, figsize = (ff*8.0, ff*6.2))
if plot_bckgrnd:
    contourf(TD_grid_m_ltp[:, :, 2], TD_grid_m_ltp[:, :, 0], TD_grid_m_ltp[:, :, 3]/TD_grid_m_ltp[:, :, 0]**3.0/T_CEP**3.0, levels = nT3_levels, cmap = cm.jet, linewidths = 5, zorder=-4)
for k in range(0, 2):
    for TD_grid_pd in TD_grids_m[::-1]:
        contourf(TD_grid_pd[:, :, 2], TD_grid_pd[:, :, 0], TD_grid_pd[:, :, 3]/TD_grid_pd[:, :, 0]**3.0/T_CEP**3.0,
                 levels = nT3_levels, cmap = cm.jet, linewidths = 5, zorder=-3)

nice_ticks()
axis([mu_min, mu_max, T_min, T_max])
colorbar(spacing = 'proportional', ticks = nT3_ticks)
plot(Tc_muT[0, :]/mu_CEP, Tc_muT[1, :]/T_CEP, color = 'grey', lw = lw_for_pd, zorder=-1)

lup_ind = 4
nT3_lvls_lower = contour(TD_grid_m[:, :, 2], TD_grid_m[:, :, 0], TD_grid_m[:, :, 3]/TD_grid_m[:, :, 0]**3.0/T_CEP**3.0,
                   levels = nT3_levels_2[:lup_ind], colors = nT3_colors_2[:lup_ind], linewidths = 3, zorder=-2)
nT3_lvls_upper = contour(TD_grid_m[:, :, 2], TD_grid_m[:, :, 0], TD_grid_m[:, :, 3]/TD_grid_m[:, :, 0]**3.0/T_CEP**3.0,
                   levels = nT3_levels_2[lup_ind:], colors = nT3_colors_2[lup_ind:], linewidths = 3, zorder=-2)

nT3_lvls_ltp = contour(TD_grid_ltp_m[:, :, 2], TD_grid_ltp_m[:, :, 0], TD_grid_ltp_m[:, :, 3]/TD_grid_ltp_m[:, :, 0]**3.0/T_CEP**3.0,
                   levels = nT3_levels_2, colors = nT3_colors_2, linewidths = 3, zorder=-2)
nT3_lvls_htp = contour(TD_grid_htp_m[:, :, 2], TD_grid_htp_m[:, :, 0], TD_grid_htp_m[:, :, 3]/TD_grid_htp_m[:, :, 0]**3.0/T_CEP**3.0,
                   levels = nT3_levels_2, colors = nT3_colors_2, linewidths = 3, zorder=-2)

# nT3_man_locs = [(200.8, 106.8), (205.5, 137.7), (57.3, 175.5), (58.8, 190.0)]
# fmts = {0.0001:'0.0001', 0.001:'0.001', 0.02:'0.02', 0.05:'0.05'}
# clabel(nT3_lvls_lower, nT3_levels_2[:lup_ind], inline = 1, fontsize = 12, fmt = fmts, manual = nT3_man_locs)

clabel(nT3_lvls_lower, nT3_levels_2[:lup_ind], inline = 1, fontsize = clabel_fs, fmt = '%1.2f', manual=set_labels_manually)
clabel(nT3_lvls_upper, nT3_levels_2[lup_ind:], inline = 1, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually) # 1.2f
scatter(1.0, 1.0, c='white', s=CEP_dotsize, alpha=1, zorder=1, edgecolor='white')

#plot(spinodals_muT['ltp'][0, :], spinodals_muT['ltp'][1, :], lw = 2, color = 'magenta')
#plot(spinodals_muT['htp'][0, :], spinodals_muT['htp'][1, :], lw = 2, color = 'yellow')


##### s/n #####
figure(3, figsize = (ff*8.0, ff*6.2))
if plot_bckgrnd:
    contourf(TD_grid_m_ltp[:, :, 2], TD_grid_m_ltp[:, :, 0], TD_grid_m_ltp[:, :, 1]/TD_grid_m_ltp[:, :, 3], levels = sn_levels, cmap = cm.jet, linewidths = 5, zorder=-4)
for k in range(0, 2):
    for TD_grid_pd in TD_grids_m[::-1]:
        contourf(TD_grid_pd[:, :, 2], TD_grid_pd[:, :, 0], TD_grid_pd[:, :, 1]/TD_grid_pd[:, :, 3], levels = sn_levels,
            cmap = cm.jet, linewidths = 5, zorder=-3)

nice_ticks()
colorbar(spacing = 'proportional', ticks = sn_ticks)
for TD_grid_pd in TD_grids_m[::-1]:
    isens = contour(TD_grid_pd[:, :, 2], TD_grid_pd[:, :, 0], TD_grid_pd[:, :, 1]/TD_grid_pd[:, :, 3],
                 levels = isen_levels, colors = isen_colors, linewidths = 3, zorder=-2)

axis([mu_min, mu_max, T_min, T_max])
plot(Tc_muT[0, :]/mu_CEP, Tc_muT[1, :]/T_CEP, color = 'grey', lw=lw_for_pd, zorder=-1)

clabel(isens, isen_levels, inline = 1, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually)
scatter(1.0, 1.0, c='white', s=CEP_dotsize, alpha=1, zorder=1, edgecolor='white')


#####   s/n  EXCERPT   #####
if plot_small_excerpt:
    figure(31, figsize = (ff*8.0, ff*6.2))
    if plot_bckgrnd:
        contourf(TD_grid_m_ltp[:, :, 2]*mu_CEP, TD_grid_m_ltp[:, :, 0]*T_CEP, TD_grid_m_ltp[:, :, 1]/TD_grid_m_ltp[:, :, 3], levels = sn_levels_excerpt, cmap = cm.jet, linewidths = 5, zorder=-4)
    for k in range(0, 2):
        for TD_grid_pd in TD_grids_m[::-1]:
            contourf(TD_grid_pd[:, :, 2]*mu_CEP, TD_grid_pd[:, :, 0]*T_CEP, TD_grid_pd[:, :, 1]/TD_grid_pd[:, :, 3], levels = sn_levels_excerpt,
                cmap = cm.jet, linewidths = 5, zorder=-3)

    nice_ticks()
    colorbar(spacing = 'proportional', ticks = sn_ticks_excerpt)
    for TD_grid_pd in TD_grids_m[::-1]:
        isens = contour(TD_grid_pd[:, :, 2]*mu_CEP, TD_grid_pd[:, :, 0]*T_CEP, TD_grid_pd[:, :, 1]/TD_grid_pd[:, :, 3],
                     levels = isen_levels_excerpt, colors = isen_colors_excerpt, linewidths = 3, zorder=-2)

    axis([0, 400, 50, 300])
    clabel(isens, isen_levels_excerpt, inline = 1, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually)


#####   s/n  EXCERPT 2   (Comparison to lattice)   #####
    fname = 'lattice/SN_lattice.p'
    SN_lattice = pickle.load(open(fname, "rb"))
    file.close(open(fname))
    levels = ['30', '51', '70', '94', '144', '420']

    figure(32, figsize = (ff*8.0, ff*6.2))
    for TD_grid_pd in TD_grids_m[::-1]:
        isens = contour(TD_grid_pd[:, :, 2]*mu_CEP, TD_grid_pd[:, :, 0]*T_CEP, TD_grid_pd[:, :, 1]/TD_grid_pd[:, :, 3],
                     levels = isen_levels_excerpt, colors = isen_colors_excerpt2, linewidths = 4, zorder=-2)
    for i in range(0, len(levels)):
        lvl = levels[i]
        plot(SN_lattice['mu_SN'+lvl], SN_lattice['T_SN'+lvl], color=isen_colors_excerpt2[i], ls='', marker='s', ms=MS, mew=MEW, zorder=-1)
    nice_ticks()
    axis([0, 400, 50, 300])
    clabel(isens, isen_levels_excerpt, inline = 1, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually, zorder=1)



##### p/T^4 #####
figure(9, figsize = (ff*8.0, ff*6.2))
for k in range(0, 2):
    #for TD_grid_pd in TD_grids_m[0:1]:
    for l in range(0, 3)[::-1]:
        TD_grid_pd = TD_grids_m[l]
        p_grid_pd = p_grids_m[l]
        contourf(TD_grid_pd[:, :, 2], TD_grid_pd[:, :, 0], p_grid_pd/TD_grid_pd[:, :, 0]**4.0/T_CEP**4.0,
                 levels = pT4_levels, cmap = cm.jet, linewidths = 5, zorder=-3)

nice_ticks()
axis([mu_min, mu_max, T_min, T_max])
colorbar(spacing = 'proportional', ticks = pT4_ticks)
plot(Tc_muT[0, :]/mu_CEP, Tc_muT[1, :]/T_CEP, color = 'grey', lw = lw_for_pd, zorder=-1)


for l in range(0, 3)[::-1]:
    TD_grid_pd = TD_grids_m[l]
    p_grid_pd = p_grids_m[l]
    pT4_lvls = contour(TD_grid_pd[:, :, 2], TD_grid_pd[:, :, 0], p_grid_pd/TD_grid_pd[:, :, 0]**4.0/T_CEP**4.0,
             levels = pT4_levels_2, colors = pT4_colors_2, linewidths = 3, zorder=-2)

clabel(pT4_lvls, pT4_levels_2, inline = 1, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually)
scatter(1.0, 1.0, c='white', s=CEP_dotsize, alpha=1, zorder=1, edgecolor='white')


#####  p  #####
figure(91, figsize = (ff*8.0, ff*6.2))
for k in range(0, 2):
    #for TD_grid_pd in TD_grids_m[0:1]:
    for l in range(0, 3)[::-1]:
        TD_grid_pd = TD_grids_m[l]
        p_grid_pd = p_grids_m[l]
        contourf(TD_grid_pd[:, :, 2], TD_grid_pd[:, :, 0], p_grid_pd,
                 levels=np.linspace(p_grid_pd.min(), p_grid_pd.max(), 300), cmap = cm.jet, linewidths = 5, zorder=-3)

nice_ticks()
axis([mu_min, mu_max, T_min, T_max])
colorbar(spacing = 'proportional')
plot(Tc_muT[0, :]/mu_CEP, Tc_muT[1, :]/T_CEP, color = 'grey', lw = lw_for_pd, zorder=-1)

div_lim = 0.02
p_levels_2 = np.hstack(( np.arange(p_grid_pd.min(), div_lim*p_grid_pd.max(), div_lim*p_grid_pd.max()/10), np.arange(div_lim*p_grid_pd.max(), p_grid_pd.max(), p_grid_pd.max()/25) ))
p_colors_2 = [None]*len(p_levels_2)
for i in range(0, len(p_colors_2)):
    p_colors_2[i] = cm.Greens(p_levels_2[i]/np.amax(p_levels_2))

for l in range(0, 3)[::-1]:
    TD_grid_pd = TD_grids_m[l]
    p_grid_pd = p_grids_m[l]
    pT4_lvls = contour(TD_grid_pd[:, :, 2], TD_grid_pd[:, :, 0], p_grid_pd,
              levels=p_levels_2, colors = p_colors_2, linewidths = 3, zorder=-2)

clabel(pT4_lvls, p_levels_2, inline = 1, fontsize = clabel_fs, fmt = '%1.1f')
scatter(1.0, 1.0, c='white', s=CEP_dotsize, alpha=1, zorder=1, edgecolor='white')


###### s/n in T-log(n/T^3) diagram ######
nT3_Tc_ltp = s_n_Tc_htpltp['ltp']['nT3']
nT3_Tc_htp = s_n_Tc_htpltp['htp']['nT3']

ltp_htp_conn_line_nT3 = np.array([np.amax(nT3_Tc_ltp), np.amin(nT3_Tc_htp)])
ltp_htp_conn_line_T = np.array([np.amax(Tc_muT[1, :])/T_CEP, np.amax(Tc_muT[1, :])/T_CEP])
print ltp_htp_conn_line_nT3, ltp_htp_conn_line_T

nT3_ptr_npts = 40
nT3_pt_region = np.zeros((len(Tc_muT[1, :]), nT3_ptr_npts))
T_pt_region = np.zeros((len(Tc_muT[1, :]), nT3_ptr_npts))
Z_pt_region = np.zeros((len(Tc_muT[1, :]), nT3_ptr_npts))
for i in range(0, len(Tc_muT[1, :])):
        nT3_pt_region[i, :] = np.linspace(nT3_Tc_ltp[i], nT3_Tc_htp[i], nT3_ptr_npts)
        T_pt_region[i, :] = np.ones(nT3_ptr_npts)*Tc_muT[1, i]/T_CEP


fig = figure(5, figsize = (ff*8.387, ff*6.55))
for k in range(0, 2):
    for TD_grid_pd in TD_grids_m[::-1]:
        contourf(np.log(TD_grid_pd[:, :, 3]/T_CEP**3.0), TD_grid_pd[:, :, 0], TD_grid_pd[:, :, 1]/TD_grid_pd[:, :, 3],
                 levels = sn_levels, cmap = cm.jet, linewidths = 5, zorder=-3)
nice_ticks()
axis([-4.0, 2.0, 0.6, T_max])
colorbar(spacing = 'proportional', ticks = sn_ticks)

plot(np.log(nT3_Tc_ltp*Tc_muT[1, :]**3.0/T_CEP**3.0), Tc_muT[1, :]/T_CEP, color = 'grey', lw = 5, zorder=0)
plot(np.log(nT3_Tc_htp*Tc_muT[1, :]**3.0/T_CEP**3.0), Tc_muT[1, :]/T_CEP, color = 'grey', lw = 5, zorder=0)
#plot(np.log(ltp_htp_conn_line_nT3*ltp_htp_conn_line_T**3.0), ltp_htp_conn_line_T, color = 'grey', lw = 5, zorder=0)
plot(np.linspace(-0.6,-0.44,50), np.ones(50), color = 'grey', lw = 4, zorder=0)
contourf(np.log(nT3_pt_region*T_pt_region**3.0), T_pt_region, Z_pt_region, levels = [-0.001, 0.001], colors = ['grey','grey'], lw = 5, zorder=0)

from matplotlib.path import Path
import matplotlib.patches as patches
verts = [
    (-1.68209, 0.71771),
    (-1.5827, 0.772012),
    (-1.327, 0.729),
    (-1.68209, 0.71771),
    ]
codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]
path = Path(verts, codes)
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, facecolor='grey', lw=4, color='grey')
ax.add_patch(patch)

for TD_grid_pd in TD_grids_m[::-1]:
    isens = contour(np.log(TD_grid_pd[:, :, 3]/T_CEP**3.0), TD_grid_pd[:, :, 0], TD_grid_pd[:, :, 1]/TD_grid_pd[:, :,3],
                    levels = isen_levels_forlog, colors = isen_colors_forlog, linewidths = 3, zorder=-2)

#axis([-8.0, 2.0, T_min, T_max])
clabel(isens, isen_levels_forlog, inline = 1, fontsize = clabel_fs, fmt = '%1.1f', manual=set_labels_manually, zorder=-1)


###### p/T^4(T_c) ######
figure(6, figsize = (ff*8.0, ff*6.0))
pT4_Tc_raster = np.zeros(len(Tc_muT[0, :]))
for i in range(0, len(pT4_Tc_raster)):
    print Tc_muT[0, i]
    pT4_Tc_raster[i] = p_Tc_data[Tc_muT[0, i]]['p_scaled']

plot(Tc_muT[1, :]/T_CEP, pT4_Tc_raster, lw = 3)
xlabel(r'$T/T_{CEP}$')
ylabel(r'$p(T_c, \mu_c)/T_c^{\,4}$')
#axis([0.72, 1.0, 1, 3])
nice_ticks()


###### p(T, mu_c(T)) ######
figure(7)
#figure(7, figsize = (ff*8.0, ff*6.0))
p_Tc_raster = np.zeros(len(Tc_muT[0, :]))
for i in range(0, len(p_Tc_raster)):
    print Tc_muT[0, i]
    p_Tc_raster[i] = p_Tc_data[Tc_muT[0, i]]['p']

plot(Tc_muT[1, :]/T_CEP, p_Tc_raster, lw = 3)
xlabel(r'$T/T_{CEP}$')
ylabel(r'$p(T, \mu_c(T)) \, [MeV^{\,4}]$')
axis([0.6, 1.0, 0.8*1e8, 1.30*1e8])
ax = subplot(111)
ax.set_xticks(np.array([0.6, 0.7, 0.8, 0.9, 1.0]))
nice_ticks()


###### p(T_c(mu), mu) ######
figure(8)
#figure(8, figsize = (ff*8.0, ff*6.0))
p_Tc_raster = np.zeros(len(Tc_muT[0, :]))
for i in range(0, len(p_Tc_raster)):
    print Tc_muT[0, i]
    p_Tc_raster[i] = p_Tc_data[Tc_muT[0, i]]['p']

plot(Tc_muT[0, :]/mu_CEP, p_Tc_raster, lw = 3)
xlabel(r'$\mu/\mu_{CEP}$')
ylabel(r'$p(T_c(\mu), \mu) \, [MeV^{\,4}]$')
axis([1.0, 1.4, 0.85*1e8, 1.30*1e8])
ax = subplot(111)
ax.set_xticks(np.array([1.0, 1.1, 1.2, 1.3, 1.4]))
ax.set_yticks(np.array([0.9, 1.0, 1.1, 1.2, 1.3])*1e8)
nice_ticks()





######

figure(1)
xlabel(r'$\mu/\mu_{CEP}$')
ylabel(r'$T/T_{CEP}$')
title(r'$s/T^{\,3}$')
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pd_sT3.pdf')

if plot_large_excerpt:
    figure(11)
    xlabel(r'$\mu/\mu_{CEP}$')
    ylabel(r'$T/T_{CEP}$')
    title(r'$s/T^{\,3}$')
    if save_figs:
        savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pd_sT3_excerpt.pdf')

figure(2)
xlabel(r'$\mu/\mu_{CEP}$')
ylabel(r'$T/T_{CEP}$')
title(r'$n/T^{\,3}$')
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pd_nT3.pdf')

figure(3)
xlabel(r'$\mu/\mu_{CEP}$')
ylabel(r'$T/T_{CEP}$')
title(r'$s/n$')
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pd_sn.pdf')

if plot_small_excerpt:
    figure(31)
    ylabel(r'$T \, [MeV\,]$')
    xlabel(r'$\mu \, [MeV\,]$')
    title(r'$s/n$')
    if save_figs:
        savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pd_sn_excerpt.pdf')

    figure(32)
    ylabel(r'$T \, [MeV\,]$')
    xlabel(r'$\mu \, [MeV\,]$')
    title(r'$s/n$')
    if save_figs:
        savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pd_sn_excerpt_lattice.pdf')

# figure(4)
# xlabel(r'$\log \, n/T^{\,3}$')
# ylabel(r'$T/T_{CEP}$')
# title(r'$s/n$')
# if save_figs:
#     savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pd_sn_nT3_T.pdf')

figure(5)
xlabel(r'$\log \, n/T_{CEP}^{\,3}$')
ylabel(r'$T/T_{CEP}$')
title(r'$s/n$')
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pd_sn_logn_T.pdf')

figure(9)
xlabel(r'$\mu/\mu_{CEP}$')
ylabel(r'$T/T_{CEP}$')
title(r'$p/T^{\, 4}$')
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pd_pT4.pdf')

figure(6)
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_pT4_Tc.pdf')

figure(7)
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_p_mucT.pdf')

figure(8)
if save_figs:
    savefig(model_type+'/'+ftype+'/pdfs/PD/'+model_type+'_p_Tcmu.pdf')

show()