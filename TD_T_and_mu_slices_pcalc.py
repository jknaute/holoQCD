import numpy as np
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from scipy.interpolate import splrep, splev
from matplotlib.transforms import Bbox
from args_and_lambds import args_dic, lambdas_dic

from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp, subplots, gcf, tight_layout, subplots_adjust, grid
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, LinearLocator

import pickle

linew = 3
rc('font', size = 20) #fontsize of axis labels (numbers)
rc('axes', labelsize = 22, lw = linew) #fontsize of axis labels (symbols)
rc('lines', mew = 2, lw = linew)
rc('patch', ec = 'k')
rc('xtick.major', pad = 7)
rc('ytick.major', pad = 7)

#rc('legend', fontsize = 20, frameon = False, fancybox = False, columnspacing = 1)
#####################
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
#T_levels = list(np.arange(50, 255, 50))
#mu_levels = list(np.arange(0, 1650, 400))

if model_type == 'G':
    fname = 'T_mu_contours_G.p'
    tmc = pickle.load(open(fname, "rb"))

    #fname2 = 'T_mu_contours_G_Tmp5.p'
    #tmcT = pickle.load(open(fname2, "rb"))
    #tmc['T'].update(tmcT['T'])

    #fname3 = 'T_mu_contours_G_moremu.p'
    #tmcmu = pickle.load(open(fname3, "rb"))
    #tmc['mu'].update(tmcmu['mu'])

    fname = 'TDTA_G.p'
    TDTA = pickle.load(open(fname, "rb"))
    file.close(open(fname))

elif model_type == 'no':
    fname = 'T_mu_contours_no.p'
    tmc = tmc = pickle.load(open(fname, "rb"))

    fname = 'TDTA_no.p'
    TDTA = pickle.load(open(fname, "rb"))
    file.close(open(fname))

print tmc['T'].keys(), tmc['mu'].keys()

T_levels = np.sort(tmc['T'].keys())
mu_levels = np.sort(tmc['mu'].keys())

#fname = 'T_mu_contours_G_moremu.p'
for key in tmc['mu'].keys():
    kk = int(key)
    print kk
    tmc['mu'].update({kk:tmc['mu'][key]})

print tmc['T'].keys(), tmc['mu'].keys()
print T_levels, mu_levels
#print buh
from TD_SB_limits2 import s_SB, rho_SB, chi_SB_F

# from fmg_plotter_Tmu_wisen6 import fig19, fig20

fig = figure(19)
fig.set_size_inches(12.0*0.8, 10.0*0.8, forward = True)
cnt1 = contour(Phi1r_raster, lphi0_raster, TD_grid[:, :, 0], levels = T_levels, cmap = cm.autumn)
cnt2 = contour(Phi1r_raster, lphi0_raster, TD_grid[:, :, 2], levels = mu_levels, cmap = cm.winter)

from plt_helper import get_contours
xy_rev = 0
path_rev = 0
rem_lr = 0
cT_contours = get_contours(cnt1, xy_rev, path_rev, T_levels, rem_lr, 'T')
print 'cTc'
if model_type == 'G':
    print cT_contours[0][0][0]
    for k in range(0, 5):
        plot(cT_contours[0][0][0][:, 0], cT_contours[0][0][0][:, 1], color = 'white', lw = 4)

ax = subplot(111)
ax.set_position(Bbox([[0.12, 0.02], [0.7, 0.9]]))
cb = colorbar(cnt2, ax = ax, orientation = 'horizontal', pad = 0.135)
cb.ax.set_xlabel(r'$\mu [MeV]$', rotation = 0)
#cb.ax.set_xlim(0, 1.0)
subplots_adjust(right = 0.80, bottom = 0.02)

ax2 = fig.add_axes([0.85, 0.108, 0.035, 0.795])
cb2 = fig.colorbar(cnt1, cax=ax2)
cb2.ax.set_ylabel(r'$T \, [MeV]$', rotation = 90)

cb2.ax.get_children()[4].set_linestyle('solid')

cb.ax.get_children()[4].set_linewidths(8)
cb2.ax.get_children()[4].set_linewidths(8)
########
cnt1_colors = [None]*len(cnt1.collections)
for lind in range(0, len(cnt1.collections)):
    cnt1_colors[lind] = cnt1.collections[lind].get_colors()[0]
cnt2_colors = [None]*len(cnt2.collections)
for lind in range(0, len(cnt2.collections)):
    cnt2_colors[lind] = cnt2.collections[lind].get_colors()[0]

from p_calcer4 import p_calc_line, p_calc_Tlvl, p_PT_calc, p_calc_mulvl

TD_Tax = TDTA[0]
phi0_raster = TDTA[1]
Phiphi_T0 = np.vstack((np.zeros(len(phi0_raster)), phi0_raster))
p_Tax = p_calc_line([TD_Tax[0,0],0],[TD_Tax[-1,0],0], TD_Tax, Phiphi_T0)
print 'p_Tax =', p_Tax
# figure(30)
# plot(TD_Tax[:,0], p_Tax[0]/TD_Tax[:,0]**4.0)
# axis([0, 500, 0, 6])
# show()

p_rasters_const_T = {}
p_rasters_const_mu = {}

for i in range(0, len(T_levels)):
    lvl = T_levels[i]
    acc_cntrTD = tmc['T'][lvl]

    TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]
    #TD_slice_scaled = acc_cntrTD[1]
    phiPhi_slice = acc_cntrTD[0]

    s_SB_raster = np.array([s_SB(lvl, mu) for mu in TD_slice_scaled[:, 2]])
    rho_SB_raster = np.array([rho_SB(lvl, mu) for mu in TD_slice_scaled[:, 2]])
    #print s_SB_raster, rho_SB_raster

    p_raster = p_calc_Tlvl(p_Tax, TD_slice_scaled, phiPhi_slice, lvl)
    print 'p_raster = ', p_raster
    p_rasters_const_T.update({lvl:p_raster})
    #pPT = p_PT_calc(p_raster[0], p_raster[1], p_raster[2])
    #print pPT

    # figure(15)
    # semilogy(p_raster[2], p_raster[0])
    # figure(16)
    # plot(p_raster[2], p_raster[0]/p_raster[2]**4.0)
    # show()
    # figure(3)
    # #plot(TD_slice_scaled[:, 2], TD_slice_scaled[:, 1]/TD_slice_scaled[:, 2]**3.0, label = r'$T = '+str(lvl)+'$ [MeV]', color = cnt1_colors[i])
    # plot(TD_slice_scaled[:, 2], TD_slice_scaled[:, 1]/s_SB_raster, label = r'$T = '+str(lvl)+'$ [MeV]', color = cnt1_colors[i])
    # #semilogy(TD_slice_scaled[:, 2], TD_slice_scaled[:, 1], label = r'$T = '+str(lvl)+'$ [MeV]', color = cnt1_colors[i])
    #
    # figure(4)
    # #plot(TD_slice_scaled[:, 2], TD_slice_scaled[:, 3]/TD_slice_scaled[:, 2]**3.0, label = r'$T = '+str(lvl)+'$ [MeV]', color = cnt1_colors[i])
    # plot(TD_slice_scaled[:, 2], TD_slice_scaled[:, 3]/rho_SB_raster, label = r'$T = '+str(lvl)+'$ [MeV]', color = cnt1_colors[i])
    # #semilogy(TD_slice_scaled[:, 2], TD_slice_scaled[:, 3], label = r'$T = '+str(lvl)+'$ [MeV]', color = cnt1_colors[i])
    # #plot(TD_slice_scaled[:, 2], TD_slice_scaled[:, 3], label = r'$T = '+str(lvl)+'$ [MeV]', color = cnt1_colors[i])

print 'Tlevels:', T_levels
for i in range(0, len(mu_levels)):
    lvl = mu_levels[i]
    acc_cntrTD = tmc['mu'][lvl]

    TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]
    #TD_slice_scaled = acc_cntrTD[1]
    phiPhi_slice = acc_cntrTD[0]
    if lvl == 0:
        p_raster = [p_Tax[0],p_Tax[0]/p_Tax[1]**4.0, p_Tax[1], TD_Tax]
    else:
        T_i = 50
        T_f = TD_slice_scaled[0, 0]
        print 'T_i, T_f:', T_i, T_f
        if T_i not in T_levels:
            print 'WARNING: T_i must match any of the T levels'

        p_Tlvl = p_rasters_const_T[T_i]
        p_raster = p_calc_mulvl([p_Tlvl[0], p_Tlvl[2]], TD_slice_scaled, phiPhi_slice, lvl, T_i, T_f)
        print 'p_list = ', p_raster[0], len(p_raster)

    p_rasters_const_mu.update({lvl:p_raster})
    #pPT = p_PT_calc(p_raster[0], p_raster[1], p_raster[2])
    #print pPT

    figure(15)
    semilogy(p_raster[2], p_raster[0])
    figure(16)
    plot(p_raster[2], p_raster[0]/p_raster[2]**4.0)
    # if lvl > 400:
    #     show()


    s_SB_raster = np.array([s_SB(T, lvl) for T in TD_slice_scaled[:, 0]])
    rho_SB_raster = np.array([rho_SB(T, lvl) for T in TD_slice_scaled[:, 0]])
    if lvl == 0:
        chi_SB_F_raster = np.array([chi_SB_F(T, lvl) for T in TD_slice_scaled[:, 0]])
        print 'chi_SB =', chi_SB_F_raster/TD_slice_scaled[:, 0]**2.0
        print 's_SB = ', s_SB_raster/TD_slice_scaled[:, 0]**3.0
    #print s_SB_raster, rho_SB_raster

    figure(1)
    #plot(TD_slice_scaled[:, 0], TD_slice_scaled[:, 1]/TD_slice_scaled[:, 0]**3.0, label = r'$\mu = '+str(lvl)+'$ [MeV]', color = cnt2_colors[i])
    plot(TD_slice_scaled[:, 0], TD_slice_scaled[:, 1]/s_SB_raster, label = r'$\mu = '+str(lvl)+'$ [MeV]', color = cnt2_colors[i])
    #semilogy(TD_slice_scaled[:, 0], TD_slice_scaled[:, 1], label = r'$\mu = '+str(lvl)+'$ [MeV]', color = cnt2_colors[i])
    #plot(TD_slice_scaled[:, 0], TD_slice_scaled[:, 1], label = r'$\mu = '+str(lvl)+'$ [MeV]', color = cnt2_colors[i])

    figure(2)
    #plot(TD_slice_scaled[:, 0], TD_slice_scaled[:, 3]/TD_slice_scaled[:, 0]**3.0, label = r'$\mu = '+str(lvl)+'$ [MeV]', color = cnt2_colors[i])
    plot(TD_slice_scaled[:, 0], TD_slice_scaled[:, 3]/rho_SB_raster, label = r'$\mu = '+str(lvl)+'$ [MeV]', color = cnt2_colors[i])
#    if lvl > 0:
#        semilogy(TD_slice_scaled[:, 0], TD_slice_scaled[:, 3], label = r'$\mu = '+str(lvl)+'$ [MeV]', color = cnt2_colors[i])


print 30*'#'
from more_TD import e_and_I, vsq, chi2_calc

fname = 'no_lattice.p'
latdata = pickle.load(open(fname, "rb"))

clrs = {0:'black', 400:'blue', 800:'green'}
lss = {0:'solid', 400:'dashed', 800:'dotted'}
for mu in [0, 400]:
    p_list = p_rasters_const_mu[mu]
    TD_n = p_list[3]
    #print TD_n[:, 0], TD_n[:, 1]
    figure(30)
    plot(p_list[2], p_list[0]/p_list[2]**4.0, lw = 2, color = clrs[mu], ls = lss[mu], label = r'$\mu = %d \, MeV$' %mu)
    if mu in latdata.keys():
        scatter(latdata[mu]['pT4'][:, 0], latdata[mu]['pT4'][:, 1], color = clrs[mu], marker = 'o', s = 34)

    figure(31)
    plot(p_list[2], TD_n[:, 1]/p_list[2]**3.0, lw = 2, color = clrs[mu], ls = lss[mu], label = r'$\mu = %d \, MeV$' %mu)

    I = e_and_I(TD_n, p_list[0])[1]

    figure(32)
    plot(p_list[2], I/p_list[2]**4.0, lw = 2, color = clrs[mu], ls = lss[mu], label = r'$\mu = %d \, MeV$' %mu)
    if mu in latdata.keys():
        scatter(latdata[mu]['IT4'][:, 0], latdata[mu]['IT4'][:, 1], color = clrs[mu], marker = 'o', s = 34)

    vsq_raster = vsq(TD_n, 'T')
    figure(33)
    plot(p_list[2], vsq_raster, lw = 2, color = clrs[mu], ls = lss[mu], label = r'$\mu = %d \, MeV$' %mu)
    if mu in latdata.keys():
        scatter(latdata[mu]['vsq'][:, 0], latdata[mu]['vsq'][:, 1], color = clrs[mu], marker = 'o', s = 34)

    figure(34)
    plot(p_list[2], TD_n[:,3]/p_list[2]**3.0, lw = 2, color = clrs[mu], ls = lss[mu], label = r'$\mu = %d \, MeV$' %mu)

    if mu > 0:
        chi2_raster = chi2_calc(TD_n)
        figure(35)
        plot(p_list[2], chi2_raster/p_list[2]**2.0, color = clrs[mu], label = r'$\mu = %d \, MeV$' %mu)

## lims for mu = 0, 400
figure(30)
axis([100, 420, 0, 4])
figure(31)
axis([100, 420, 0, 15.5])
figure(32)
axis([100, 420, 0, 6])
figure(33)
axis([100, 405, 0, 0.4])
MLs = {30:[50, 10, 1, 0.2], 31:[50, 10, 5, 1], 32:[50, 10, 1, 0.2], 33:[50, 10, 0.1, 0.02]}

## lims for mu = 0, 400, 800
# figure(30)
# axis([50, 420, 0, 4.5])
# figure(31)
# axis([50, 420, 0, 18.5])
# figure(32)
# axis([50, 420, 0, 18.5])
# figure(33)
# axis([50, 420, 0, 0.4])
# MLs = {30:[50, 10, 1, 0.2], 31:[50, 10, 5, 1], 32:[50, 10, 5, 1], 33:[50, 10, 0.1, 0.02]}

ylbls = {30:r'$p/T^{\,4}$', 31:r'$s/T^{\,3}$', 32:r'$I/T^{\,4}$', 33:r'$v_s^2$'}
savenames = {30:'pT4', 31:'sT3', 32:'IT4', 33:'vsq'}

for fig in [30, 31, 32, 33]:
    figure(fig)
    xlabel(r'$T \, [MeV\,]$')
    ylabel(ylbls[fig])
    ax = subplot(111)
    ax.xaxis.set_major_locator(MultipleLocator(MLs[fig][0]))
    ax.xaxis.set_minor_locator(MultipleLocator(MLs[fig][1]))

    ax.yaxis.set_major_locator(MultipleLocator(MLs[fig][2]))
    ax.yaxis.set_minor_locator(MultipleLocator(MLs[fig][3]))
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(10)
        l.set_markeredgewidth(2.0)
    for l in ax.yaxis.get_minorticklines() + ax.xaxis.get_minorticklines():
        l.set_markersize(5)
        l.set_markeredgewidth(1.5)

    ax.set_position(Bbox([[0.14, 0.14], [0.95, 0.95]]))
    legend(loc = 'best', fontsize = 18, frameon = False)
    #grid(which = 'both')
    #savefig('no_'+savenames[fig]+'.pdf')
show()


axlim1 = [0, 600.0, -0.3, 50.0]
axlim2 = [0, 600.0, -7.0/150.0, 7.0]
axlim3 = [0, 3000.0, -0.2/150.0, 0.2]
axlim4 = [0, 3000.0, -0.02/150.0, 0.02]

axlims = [axlim1, axlim2, axlim3, axlim4]
ylbls = [r'$s/T^{\,3}$', r'$\rho/T^{\,3}$', r'$s/\mu^3$', r'$\rho/\mu^3$']
lglocs = ['upper right', 'best', 'upper right', 'upper right']

axlim1SBs = [0, 600.0, -0.01, 1.0]
axlim2SBs = [0, 600.0, -0.01, 1.0]
axlim3SBs = [0, 3000.0, -0.01, 1.0]
axlim4SBs = [0, 3000.0, -0.01, 1.0]

axlimsSBs = [axlim1SBs, axlim2SBs, axlim3SBs, axlim4SBs]
ylblsSBs = [r'$s/s_{SB}$', r'$\rho/\rho_{SB}$', r'$s/s_{SB}$', r'$\rho/\rho_{SB}$']

lglocsSBs = ['upper right', 'best', 'upper right', 'upper right']

figure(1)
xlabel(r'$T \,[MeV]$')

figure(2)
xlabel(r'$T \,[MeV]$')

figure(3)
xlabel(r'$\mu [MeV]$')

figure(4)
xlabel(r'$\mu [MeV]$')

ffac = 1.0
for i in range(1, 5):
    figure(i, figsize = (ffac*0.8, ffac*0.6))
    ax = subplot(111)
    #ax = subplot(111)
    axis(axlimsSBs[i - 1])
    ylabel(ylblsSBs[i - 1])
    #legend(fontsize = 16, labelspacing = 0.03, loc = lglocsSBs[i - 1])
    ax.set_position(Bbox([[0.17, 0.14], [0.95, 0.95]]))
    for l in ax.get_xticklines() + ax.get_yticklines():
        l.set_markersize(8)
        l.set_markeredgewidth(3)

    #savefig('fig%d.pdf' %i)


############################
f, ((ax1, ax2), (ax3, ax4)) = subplots(2, 2, sharex='col', sharey='row')
f.set_size_inches(12.0, 10.0, forward=True)
#ax = subplot(111)

for i in range(0, len(T_levels)):
    lvl = T_levels[i]
    acc_cntrTD = tmc['T'][lvl]

    TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]
    ax2.semilogy(TD_slice_scaled[:, 2]*0.1, TD_slice_scaled[:, 1], color = cnt1_colors[i], label = r'$T = '+str(lvl)+'$ [MeV]')
    ax4.semilogy(TD_slice_scaled[:, 2]*0.1, TD_slice_scaled[:, 3], color = cnt1_colors[i], label = r'$T = '+str(lvl)+'$ [MeV]')

for i in range(0, len(mu_levels)):
    lvl = mu_levels[i]
    acc_cntrTD = tmc['mu'][lvl]

    TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]
    ax1.semilogy(TD_slice_scaled[:, 0], TD_slice_scaled[:, 1], color = cnt2_colors[i], label = r'$\mu = '+str(lvl)+'$ [MeV]')
    if lvl > 0:
        ax3.semilogy(TD_slice_scaled[:, 0], TD_slice_scaled[:, 3], color = cnt2_colors[i])

ax1.set_xlim([0, 300])
ax2.set_xlim([0, 300])
tight_layout()

#
ax1.set_ylabel(r'$s [MeV]^3$')

ax3.set_xlabel(r'$T \, [MeV]$')
ax3.set_ylabel(r'$\rho [MeV]^3$')

ax4.set_xlabel(r'$0.1 \mu [MeV]$')
ax1.set_ylim([1e-1, 2*1e9])

#ax1.legend(loc = 'lower right', fontsize = 16, labelspacing = 0.3)
#ax2.legend(loc = 'lower center', fontsize = 16, ncol = 2, labelspacing = 0.3)

tight_layout()
subplots_adjust(left = 0.12, bottom = 0.1)

axs = [ax1, ax2, ax3, ax4]
for axi in axs:
    for l in axi.get_xticklines() + axi.get_yticklines():
        l.set_markersize(8)
        l.set_markeredgewidth(2)


show()
