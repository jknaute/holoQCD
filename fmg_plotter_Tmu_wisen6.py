import numpy as np
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from scipy.interpolate import splrep, splev
from args_and_lambds import args_dic, lambdas_dic

from pylab import * # figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot

import pickle
from matplotlib.transforms import Bbox
from nice_tcks import nice_ticks

linew = 3
rc('font', size = 20) #fontsize of axis labels (numbers)
rc('axes', labelsize = 22, lw = linew) #fontsize of axis labels (symbols)
rc('lines', mew = 2, lw = linew, markeredgewidth = 2)
rc('patch', ec = 'k')
rc('xtick.major', pad = 7)
rc('ytick.major', pad = 7)
rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'
rcParams['figure.figsize'] = [8.0, 6.0]
#####################

model_type = 'VRY_4'
ftype = args_dic['ftype'][model_type]

fname = model_type+'/'+ftype+'/TD_gr_'+model_type+'_wmu0.p'
TD_gr = pickle.load(open(fname, "rb"))
lambdas = lambdas_dic[model_type]

TD_grid = TD_gr[0]
phiPhi_grid = TD_gr[1]
TD_full = TD_gr[2]

phi0_pts = TD_full['phi0_pts']
Phi1_pts = TD_full['Phi1_pts']

TD_grid_s = TD_scale(TD_grid, lambdas)[0]


plot_const_lines = 1        # draw lines of constant phi0 and Phi1
interval_for_phi0_lines = 1 # select the interval for drawn phi0=const lines


figure(1)
for i in range(0, phi0_pts):
    if plot_const_lines and i%interval_for_phi0_lines==0:
        plot(phiPhi_grid[i, :, 1], phiPhi_grid[i, :, 0], '-c', lw=1, zorder=-1)
    for j in range(0, Phi1_pts):
        if TD_grid_s[i, j, 0] != 0 and TD_grid_s[i, j, 0] < 1e5:
             scatter(phiPhi_grid[i, j, 1], phiPhi_grid[i, j, 0], color = 'red', s = 16, zorder=1)
if plot_const_lines:
    plot(phiPhi_grid[:, :, 1], phiPhi_grid[:, :, 0], '-m', lw=1, zorder=-1)
xlabel(r'$\Phi_1/\Phi_1^{max}$')
ylabel(r'$\phi_0$')
nice_ticks()


figure(2)
for i in range(0, phi0_pts):
    if plot_const_lines and i%interval_for_phi0_lines==0:
        plot(TD_grid_s[i, :, 2], TD_grid_s[i, :, 0], '-c', lw=1, zorder=-1)
    for j in range(0, Phi1_pts):
        if TD_grid_s[i, j, 0] != 0 and TD_grid_s[i, j, 0] < 1e5:
            scatter(TD_grid_s[i, j, 2], TD_grid_s[i, j, 0], color = 'red', s = 16, zorder=1)
if plot_const_lines:
    plot(TD_grid_s[:, :, 2], TD_grid_s[:, :, 0], '-m', lw=1, zorder=-1)
axis([0, 5000, 0, 500])
xlabel(r'$\mu \, [MeV\,]$')
ylabel(r'$T \, [MeV\,]$')
nice_ticks()


figure(3)
for i in range(0, phi0_pts):
    if i%interval_for_phi0_lines==0:
        plot(TD_grid_s[i, :, 3], TD_grid_s[i, :, 0], '-c', lw=1, zorder=-1)
    for j in range(0, Phi1_pts):
        if TD_grid_s[i, j, 0] != 0 and TD_grid_s[i, j, 0] < 1e5:
            scatter(TD_grid_s[i, j, 3], TD_grid_s[i, j, 0], color = 'red', s = 16, zorder=1)
plot(TD_grid_s[:, :, 3], TD_grid_s[:, :, 0], '-m', lw=1, zorder=-1)
xlabel(r'$n \, [MeV^{\,3}]$')
ylabel(r'$T \, [MeV\,]$')
nice_ticks()


figure(4)
for i in range(0, phi0_pts):
    if i%interval_for_phi0_lines==0:
        plot(TD_grid_s[i, :, 1], TD_grid_s[i, :, 0], '-c', lw=1, zorder=-1)
    for j in range(0, Phi1_pts):
        if TD_grid_s[i, j, 0] != 0 and TD_grid_s[i, j, 0] < 1e5:
            scatter(TD_grid_s[i, j, 1], TD_grid_s[i, j, 0], color = 'red', s = 16, zorder=1)
plot(TD_grid_s[:, :, 1], TD_grid_s[:, :, 0], '-m', lw=1, zorder=-1)
xlabel(r'$s \, [MeV^{\,3}]$')
ylabel(r'$T \, [MeV\,]$')
nice_ticks()


show()




#####################
# fname = 'TD_gr_G_isen3.p'
# TD_isen_dic = pickle.load(open(fname, "rb"))
# TD_grisen = TD_gr_isen_full[0]

J = J_calc_fd(TD_grid, phi0_pts, Phi1_pts, phiPhi_grid)
print 'buh'
#TD_grisen_s = TD_scale(TD_grisen, lambdas)[0]

def postsort(axs_list, sortax):
    axnum = len(axs_list)
    argmin = axs_list[sortax].argmin()
    newargs = np.argsort(axs_list[sortax][:argmin])[::-1]
    for ax in range(0, axnum):
        axs_list[ax][:argmin] = axs_list[ax][:argmin][newargs]
        axs_list[ax][argmin:] = axs_list[ax][argmin:]

    return axs_list

def get_contour(TD_grid, J, phi0_pts, Phi1_pts, phiPhi_grid, phase_type, ps):
    contour_bool = np.zeros_like(J, dtype = bool)
    for i in range(1, phi0_pts):
        for j in range(0, Phi1_pts):
            #print i, j, J[i, j], J[i - 1, j]
            if J[i, j]*J[i - 1, j] < 0 and J[i - 1, j] > 0:
                if phase_type == 'unst':
                    contour_bool[i, j] = 1
                elif phase_type == 'st':
                    contour_bool[i - 1, j] = 1
            elif J[i, j]*J[i - 1, j] < 0 and J[i - 1, j] < 0:
                if phase_type == 'unst':
                    contour_bool[i - 1, j] = 1
                elif phase_type == 'st':
                    contour_bool[i, j] = 1
            if i == phi0_pts - 1 and J[i, j] < 0:
                contour_bool[i, j] = 1

    c_phiPhi = phiPhi_grid[contour_bool]
    c_TD = TD_grid[contour_bool]

    #print postsort, postsort[1][0]#, postsort[0]
    print c_phiPhi
    if ps[0]:
        phiPhi_psort = postsort([c_phiPhi[:, 0], c_phiPhi[:, 1]], ps[1][0])
        c_phiPhi_srt = np.zeros_like(c_phiPhi)
        for k in range(0, len(phiPhi_psort)):
            c_phiPhi_srt[:, k] = phiPhi_psort[k]#[::-1]
    else:
        c_phiPhi_srt = c_phiPhi

    #print c_phiPhi_srt
    if ps[0]:
        TD_psort = postsort([c_TD[:, 0], c_TD[:, 1], c_TD[:, 2], c_TD[:, 3]], ps[1][1])
        c_TD_srt = np.zeros_like(c_TD)
        for k in range(0, len(TD_psort)):
            c_TD_srt[:, k] = TD_psort[k]
    else:
        c_TD_srt = c_TD

    return [contour_bool, c_phiPhi_srt, c_TD_srt]


#cont = get_contour(TD_grid_s, J, phi0_pts, Phi1_pts, phiPhi_grid, 'st', [1, [1, 2]])
### last arg is postsort parameters: [postsort?,[phiPhi_ax, TD_ax]]
#print cont[1]
#print cont[2]
##print cont[1][:, 0]

#CEP_ind = cont[2][:, 2].argmin() #index of smallest unstable mu value
#CEP_loc_TD = cont[1][CEP_ind, :]
#CEP_loc_phiPhi = cont[2][CEP_ind, :]
#print CEP_loc_phiPhi
#print CEP_loc_TD
#CEP_coll = [CEP_ind, CEP_loc_phiPhi, CEP_loc_TD]
#print 'CEP_coll: ', CEP_coll


def get_mixedphase_stable_pts(TD_grid, phi0_pts, Phi1_pts, cont, CEP_coll):
    mps = np.zeros((phi0_pts, Phi1_pts))
    CEP_ind = CEP_coll[0]
    CEP_loc_TD = CEP_coll[1]
    #print CEP_loc_TD
    l_mpb = [cont[2][:, 0][:CEP_ind], cont[2][:, 2][:CEP_ind]] #T, mu left mixed phase boundary
    r_mpb = [cont[2][:, 0][CEP_ind:], cont[2][:, 2][CEP_ind:]] #T, mu right mixed phase boundary

    l_mpb_tck = splrep(l_mpb[1][::-1], l_mpb[0][::-1])
    r_mpb[1], r_mpb[0] = zip(*sorted(zip(r_mpb[1], r_mpb[0])))
    r_mpb_tck = splrep(r_mpb[1], r_mpb[0])
    for i in range(0, phi0_pts):
        for j in range(0, Phi1_pts):
            T = TD_grid[i, j, 0]
            mu = TD_grid[i, j, 2]
            if T <= CEP_loc_TD[0] and mu >= CEP_loc_TD[2] and mu <= r_mpb[1][-1]:
                if mu <= l_mpb[1][0]:
                    T_l_projection = splev(mu, l_mpb_tck)
                    T_r_projection = splev(mu, r_mpb_tck)
                    if T >= T_l_projection and T <= T_r_projection*(1.0 + 2.0*1e-3):
                        ##(1.0 + 2*1e-3) is to account for finite accuracy of mpb determination
                        mps[i, j] = 1
                elif mu > l_mpb[1][0]:
                    T_r_projection = splev(mu, r_mpb_tck)
                    if T <= T_r_projection*(1.0 + 2.0*1e-3):
                        mps[i, j] = 1

    return mps

def mps_J_like(mps, phi0_pts, Phi1_pts):
    mps_J = np.zeros_like(mps)
    for i in range(0, phi0_pts):
        for j in range(0, Phi1_pts):
            if mps[i, j] == 0:
                mps_J[i, j] = 1
            elif mps[i, j] == 1:
                mps_J[i, j] = -1
    return mps_J


def get_phiPhi_mps_contour(J, phi0_pts, Phi1_pts, phiPhi_grid, phase_type):
    contour_bool = np.zeros_like(J, dtype = bool)
    for j in range(0, Phi1_pts):
        for i in range(1, phi0_pts):
            print i, j, phiPhi_grid[i, j], J[i, j], J[i - 1, j]
            if J[i, j]*J[i - 1, j] < 0 and J[i - 1, j] > 0:
                if phase_type == 'unst':
                    contour_bool[i, j] = 1
                elif phase_type == 'st':
                    contour_bool[i - 1, j] = 1
                break

    c_phiPhi = phiPhi_grid[contour_bool]

    phiPhi_psort = postsort([c_phiPhi[:, 0], c_phiPhi[:, 1]], 1)
    c_phiPhi_srt = np.zeros_like(c_phiPhi)
    for k in range(0, len(phiPhi_psort)):
        c_phiPhi_srt[:, k] = phiPhi_psort[k]#[::-1]

    return [contour_bool, c_phiPhi_srt]

#mps = get_mixedphase_stable_pts(TD_grid_s, phi0_pts, Phi1_pts, cont, CEP_coll)
#mps_J = mps_J_like(mps, phi0_pts, Phi1_pts)
#mps_cont = get_phiPhi_mps_contour(mps_J, phi0_pts, Phi1_pts, phiPhi_grid, 'unst')
#print mps_cont[1]
#print 4*'#'

#figure(10)
#plot(TD_grid_s[:, 0, 0], TD_grid_s[:, 0, 1]/TD_grid_s[:, 0, 0]**3.0)

i_samp1 = 20
i_samp2 = 36
i_samp3 = 48
i_samp4 = 55

j_samp1 = 10
j_samp2 = 20
j_samp3 = 30
j_samp4 = 40
j_samp5 = 45

i_samps = [i_samp1, i_samp2, i_samp3]
j_samps = [j_samp1, j_samp2, j_samp3]

mrkdic_i = {} # {i_samp1:'s', i_samp2:'d', i_samp3:'D', i_samp4:'s'}
mrkdic_j = {} # {j_samp1:'^', j_samp2:'o', j_samp3:'*', j_samp4:'^', j_samp5:'o'}


plot_phiPhilines = 1
plot_isentropes = 0
plot_unst_contour = 0
plot_mps_contour = 0
plot_mps = 0

print phi0_pts, Phi1_pts

for i in range(0, phi0_pts):
    for j in range(0, Phi1_pts):
        clr = 'blue'
        if J[i, j] > 0:
            clr = 'red'
        elif J[i, j] < 0:
            clr = 'green'
        mrkr = 'o'
        size = 5
        lw = 1
        mew = 0
        zorder = -1
        if plot_phiPhilines:
            if i in mrkdic_i.keys():
                mrkr = mrkdic_i[i]
                size = 6
                lw = 1.5
                mew = 1
                zorder = 0
            if j in mrkdic_j.keys():
                mrkr = mrkdic_j[j]
                size = 6
                if mrkr == '*':
                    size = 7
                lw = 1.5
                mew = 1
                zorder = 1

        if plot_mps:
            if J[i, j] > 0:
                if mps[i, j]:
                    clr = 'magenta'
        ###
        figure(19) # phi0 - Phi1 plane
        if TD_grid_s[i, j, 0] != 0:
            if plot_phiPhilines:
                if plot_const_lines and i%interval_for_phi0_lines == 0:
                    plot(phiPhi_grid[i, :, 1], phiPhi_grid[i, :, 0], '-c', lw=1, zorder=-2)
                plot(phiPhi_grid[i, j, 1], phiPhi_grid[i, j, 0], color = clr, marker = mrkr, markersize = size, lw = lw, mew = mew, zorder = zorder)
            else:
                scatter(phiPhi_grid[i, j, 1], phiPhi_grid[i, j, 0], color = clr, s = 16)
        ###
        figure(20) # T - mu plane
        if TD_grid_s[i, j, 0] != 0 and TD_grid_s[i, j, 0] < 1e3 and TD_grid_s[i, j, 0] < 1e5:
            if plot_phiPhilines:
                if plot_const_lines and i%interval_for_phi0_lines == 0:
                    plot(TD_grid_s[i, :, 2], TD_grid_s[i, :, 0], '-c', lw=1, zorder=-2)
                plot(TD_grid_s[i, j, 2], TD_grid_s[i, j, 0], color = clr, marker = mrkr, markersize = size, lw = lw, mew = mew, zorder = zorder)
            else:
                scatter(TD_grid_s[i, j, 2], TD_grid_s[i, j, 0], color = clr, s = 16)

if plot_unst_contour:
    figure(19)
    plot(cont[1][:, 1], np.log(cont[1][:, 0]), color = 'green', alpha = 0.5)
    figure(20)
    plot(cont[2][:, 2], cont[2][:, 0], color = 'green', alpha = 0.5)

if plot_mps_contour:
    figure(19)
    plot(mps_cont[1][:, 1], np.log(mps_cont[1][:, 0]), color = 'magenta', alpha = 0.5)
    #figure(20)
    #plot(mps_cont[2][:, 2], mps_cont[2][:, 0])


figure(19)
if plot_const_lines:
    plot(phiPhi_grid[:, :, 1], phiPhi_grid[:, :, 0], '-m', lw=1, zorder=-2)
xlabel(r'$\Phi_1/\Phi_1^{max}$')
ylabel(r'$\phi_0$')
#axis([0, 1.0, 0, 2.5])
nice_ticks()
savefig(model_type+'/'+ftype+'/pdfs/phi_Phi_J_'+model_type+'.pdf')


figure(20)
if plot_const_lines:
    plot(TD_grid_s[:, :, 2], TD_grid_s[:, :, 0], '-m', lw=1, zorder=-2)
xlabel(r'$\mu \, [MeV\,]$')
ylabel(r'$T \, [MeV\,]$')
axis([0, 1500.0, 0, 150.0])
nice_ticks()
savefig(model_type+'/'+ftype+'/pdfs/T_mu_J_'+model_type+'.pdf')


#ax = subplot(111)
#ax.set_position(Bbox([[0.14, 0.14], [0.95, 0.95]]))
#for l in ax.get_xticklines() + ax.get_yticklines():
    #l.set_markersize(8)
    #l.set_markeredgewidth(linew)

if plot_isentropes:
    isen_clrs = ['blue', 'grey', 'olive', 'purple', 'green']
    j = 0
    epbs = np.sort(TD_isen_dic.keys())
    for epb in epbs:
        print 'epb =', epb
        TD_isen_s = TD_scale_isen(TD_isen_dic[epb][0], lambdas)[0]
        phiPhi_isen = TD_isen_dic[epb][1]
        #phiPhi_isen[:, 1]
        T_raster = TD_isen_s[:, 0]
        print len(T_raster), len(TD_isen_s[:, 2]), len(phiPhi_isen[:, 1]), len(phiPhi_isen[:, 1])

        Phi1r_raster = np.compress(T_raster > 0, phiPhi_isen[:, 1])[:-1]
        phi0_raster = np.compress(T_raster > 0, phiPhi_isen[:, 0])[:-1]
        mu_raster = np.compress(T_raster > 0, TD_isen_s[:, 2])[:-1]
        T_raster = np.compress(T_raster > 0, TD_isen_s[:, 0])[:-1]

        figure(19)
        plot(Phi1r_raster, np.log(phi0_raster), color = isen_clrs[j], label = r'$s/n = %2.1f$' %epb)
        figure(20)
        plot(mu_raster, T_raster, color = isen_clrs[j], label = r'$s/n = %2.1f$' %epb)
        j += 1

#figure(19)
#legend(loc = 'upper left', ncol = 1)
#figure(20)
#legend(loc = 'lower left', ncol = 1)

#fig19 = figure(19)
#fig20 = figure(20)
#fig19.set_size_inches(12.0*0.8, 10.0*0.8, forward = True)
#fig20.set_size_inches(12.0*0.8, 10.0*0.8, forward = True)

show()
