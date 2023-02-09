import numpy as np
from fmg_TDprocess import TD_scale, TD_scale_isen, J_calc_fd
from scipy.interpolate import splrep, splev
from matplotlib.transforms import Bbox
from args_and_lambds import args_dic, lambdas_dic

from pylab import * #figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp, subplots, gcf, tight_layout, subplots_adjust, grid, ticklabel_format
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, LinearLocator

import pickle
from nice_tcks import nice_ticks
from TD_SB_limits2 import s_SB, rho_SB, chi_SB_F
#from fmg_plotter_Tmu_wisen6 import fig19, fig20
from p_calcer4 import p_calc_line, p_calc_Tlvl, p_PT_calc, p_calc_mulvl
from plt_helper import get_contours
from more_TD import e_and_I, vsq, chi2_calc

linew = 3
rc('font', size = 20) #fontsize of axis labels (numbers)
rc('axes', labelsize = 22, lw = linew) #fontsize of axis labels (symbols)
rc('lines', mew = 2, lw = linew)
rc('patch', ec = 'k')
rc('xtick.major', pad = 7)
rc('ytick.major', pad = 7)
#rc('legend', fontsize = 20, frameon = False, fancybox = False, columnspacing = 1)
rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'
rcParams['figure.figsize'] = [8.0, 6.0]

MS = 5          # marker size
MEW = 1         # marker edge width
lfs = 20        # legend font size


#####################
model_type = 'VRY_4'
ftype = args_dic['ftype'][model_type]

V_args = args_dic['V'][model_type]
f_args = args_dic['f'][model_type]
lambdas = lambdas_dic[model_type]

plot_PDstuff = 0    # Tc(mu_c) and contour lines
#####################


fname = model_type+'/'+ftype+'/TD_gr_'+model_type+'_wmu0.p'
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
    fname = 'T_mu_contours_G_copy.p'
    tmc = pickle.load(open(fname, "rb"))
    file.close(open(fname))

    # fname2 = 'T_mu_contours_G_moremore_mu.p'
    # tmc2 = pickle.load(open(fname2, "rb"))
    # file.close(open(fname2))
    # print tmc2.keys()
    # tmc['mu'].update(tmc2['mu'])

    fname = 'TDTA_G.p'
    TDTA = pickle.load(open(fname, "rb"))
    file.close(open(fname))

elif model_type == 'no':
    fname = model_type+'/'+ftype+'/T_mu_contours_no.p'
    tmc = tmc = pickle.load(open(fname, "rb"))

    fname = model_type+'/'+ftype+'/TDTA_no.p'
    TDTA = pickle.load(open(fname, "rb"))
    file.close(open(fname))

elif model_type == 'VRY_2':
    fname = model_type+'/'+ftype+'/T_mu_contours_'+model_type+'.p'
    tmc = tmc = pickle.load(open(fname, "rb"))

    fname = model_type+'/'+ftype+'/TDTA_'+model_type+'.p'
    TDTA = pickle.load(open(fname, "rb"))
    file.close(open(fname))

    fname = model_type+'/'+ftype+'/p_on_levels_'+model_type+'.p'
    p_on_lvls = pickle.load(open(fname, 'rb'))
    file.close(open(fname))

elif model_type == 'VRY_4':
    fname = model_type+'/'+ftype+'/T_mu_contours_'+model_type+'.p'
    tmc = tmc = pickle.load(open(fname, "rb"))

    fname = model_type+'/'+ftype+'/TDTA_'+model_type+'.p'
    TDTA = pickle.load(open(fname, "rb"))
    file.close(open(fname))


print tmc['T'].keys(), tmc['mu'].keys()

#tmc['mu'].update(tmc2['mu'])
#pickle.dump(tmc, open('T_mu_contours_G.p', "wb"))
#file.close(open('T_mu_contours_G.p'))

T_levels = np.sort(tmc['T'].keys())
mu_levels = np.sort(tmc['mu'].keys())
if model_type=='VRY_4':
    mu_levels = np.array([0, 200, 300, 400, 500, 600, 610, 611, 612, 650, 700, 800, 900])
print 'T_levels: ', T_levels
print 'mu_levels: ', mu_levels
#print buh

#for key in tmc['mu'].keys():
    #kk = int(key)
    #print kk
    #tmc['mu'].update({kk:tmc['mu'][key]})

#print tmc['T'].keys(), tmc['mu'].keys()
#print T_levels, mu_levels
#print buh

### lattice TD at mu_B = 400 MeV (WuB, O(mu^2), 2012):
fname = 'lattice/TD_lattice_mu400.p'
TD_lattice_mu400 = pickle.load(open(fname, "rb"))
file.close(open(fname))


if plot_PDstuff:
    fig = figure(19)
    fig.set_size_inches(12.0*0.8, 10.0*0.8, forward = True)
    cnt1 = contour(Phi1r_raster, lphi0_raster, TD_grid[:, :, 0], levels = T_levels, cmap = cm.autumn)
    cnt2 = contour(Phi1r_raster, lphi0_raster, TD_grid[:, :, 2], levels = mu_levels, cmap = cm.winter)


    #if model_type == 'no':
        #xy_rev = 0
        #path_rev = 0
        #rem_lr = 0
    if model_type=='no' or model_type=='VRY_2':
        xy_rev = 0
        path_rev = 0
        rem_lr = 0
    cT_contours = get_contours(cnt1, xy_rev, path_rev, T_levels, rem_lr, 'T')
    print 'cTc'
    # if model_type == 'G':
    #     print cT_contours[0][0][0]
    #     for k in range(0, 5):
    #         plot(cT_contours[0][0][0][:, 0], cT_contours[0][0][0][:, 1], color = 'white', lw = 4)
    #         #plot(cT_contours[0][0][:, 0], cT_contours[0][0][:, 1], color = 'white', lw = 4)

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


TD_Tax = TDTA[0]
phi0_raster = TDTA[1]
Phiphi_T0 = np.vstack((np.zeros(len(phi0_raster)), phi0_raster))
p_Tax = p_calc_line([TD_Tax[0,0],0],[TD_Tax[-1,0],0], TD_Tax, Phiphi_T0)
print 'p_Tax =', p_Tax
print 'Tax: ', TD_Tax[:,0]
#figure(30)
#plot(TD_Tax[:,0], p_Tax[0]/TD_Tax[:,0]**4.0)
#axis([0, 500, 0, 6])
#show()

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
    print 'T, p0 = ', lvl, p_raster[0][0]/lvl**4.0
    p_rasters_const_T.update({lvl:p_raster})

    #figure(i)
    #plot(p_raster[2], p_raster[0]/lvl**4.0)
    #show()


### pressure adjustment along mu=const levels to pT4(mu, T=T_adjust) for integration constant:
if model_type=='VRY_2':
    T_adjust = 125.0
    pT4_Tlvl_adjust_tck = splrep(p_on_lvls['p_along_T'][T_adjust][2], p_on_lvls['p_along_T'][T_adjust][0]/T_adjust**4)

#figure(100)
#plot(p_on_lvls['p_along_T'][T_adjust][2], p_on_lvls['p_along_T'][T_adjust][0]/T_adjust**4)
#plot(p_on_lvls['p_along_T'][T_adjust][2], splev(p_on_lvls['p_along_T'][T_adjust][2], pT4_Tlvl_adjust_tck), ls='--')
#show()

for i in range(0, len(mu_levels)): #[:-1]:
    lvl = mu_levels[i]
    print 'mu_lvl = ', lvl
    acc_cntrTD = tmc['mu'][lvl]

    TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]
    #TD_slice_scaled = acc_cntrTD[1]
    phiPhi_slice = acc_cntrTD[0]
    if lvl == 0:
        p_raster = [p_Tax[0], p_Tax[0]/p_Tax[1]**4.0, p_Tax[1], TD_Tax]
    else:
        T_i = 55 # 50
        T_f = TD_slice_scaled[0, 0]
        print 'mu_lvl, T_i, T_f:', lvl, T_i, T_f
        if T_i not in T_levels:
            print 'WARNINING: T_i must match any of the T levels'

        p_Tlvl = p_rasters_const_T[T_i]
        print 'p_Tlvl: ', p_Tlvl
        p_raster = p_calc_mulvl([p_Tlvl[0], p_Tlvl[2]], TD_slice_scaled, phiPhi_slice, lvl, T_i, T_f)
        print 'p_list = ', p_raster[0], len(p_raster)

    ## adjustment:
    if lvl>0 and model_type=='VRY_2':
        Tlarge_inds = np.where(p_raster[2]>120.0)[0]                                    # select temp. range above 1st order PT to avoid spline problems in multivalued region
        pT4_mulvl_adjust_tck = splrep(p_raster[2][Tlarge_inds], p_raster[0][Tlarge_inds]/p_raster[2][Tlarge_inds]**4.0) # = pT4(mulvl, T)
        diff = splev(lvl, pT4_Tlvl_adjust_tck) - splev(T_adjust, pT4_mulvl_adjust_tck)  # soll - ist
        #if diff>0:
            #p_raster[0] = p_raster[0] + diff*T_adjust**4
            #p_raster[1] = p_raster[1] + diff
        #if diff<0:
            #p_raster[0] = p_raster[0] - diff*T_adjust**4
            #p_raster[1] = p_raster[1] - diff
        print 'mu, Tlarge_inds = ', lvl, Tlarge_inds
        print 'soll, ist, diff = ', splev(lvl, pT4_Tlvl_adjust_tck), splev(T_adjust, pT4_mulvl_adjust_tck), diff
        #figure(i)
        #plot(p_raster[2][Tlarge_inds], p_raster[1][Tlarge_inds])
        #show()

    p_rasters_const_mu.update({lvl:p_raster})

    s_SB_raster = np.array([s_SB(T, lvl) for T in TD_slice_scaled[:, 0]])
    rho_SB_raster = np.array([rho_SB(T, lvl) for T in TD_slice_scaled[:, 0]])
    if lvl == 0:
        chi_SB_F_raster = np.array([chi_SB_F(T, lvl) for T in TD_slice_scaled[:, 0]])
        print 'chi_SB =', chi_SB_F_raster/TD_slice_scaled[:, 0]**2.0
        print 's_SB = ', s_SB_raster/TD_slice_scaled[:, 0]**3.0
    #print s_SB_raster, rho_SB_raster

print 30*'#'


#if model_type == 'no':
    #mulist = [0, 400]
    #clrs = {0:'black', 400:'blue', 800:'green'}
    #clrs2 = {0:'orange', 400:'red', 800:'green'}
    #lss = {0:'solid', 400:'dashed', 800:'dotted'}

if model_type=='no' or model_type=='VRY_2' or model_type=='VRY_4':
    clr_list_0 = ['black', 'C3', 'blue', 'orange', 'green', 'magenta', 'crimson','grey']
    lss_list_0 = ['solid']
    #mulist = [0, 2, 200, 400, 600, 700, 800, 900, 1000, 1200, 1400]

    clrs = [None]*len(mu_levels)
    lss = [None]*len(mu_levels)
    for i in range(0, len(clrs)):
        clrs[i] = clr_list_0[i % len(clr_list_0)]
        lss[i] = lss_list_0[i % len(lss_list_0)]

print clrs, lss

if model_type == 'VRY_2':
    mulist = mu_levels[:-1]
    mulist = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 990, 1200]
    mu_plotlist = [0, 400, 600, 800, 990, 1200]
if model_type == 'no':
    mulist = [400.]
    mu_plotlist = [400.]
    nocurves=pickle.load(open('no_curves.p','rb')) # paper read out
if model_type == 'VRY_4':
    mulist = [0, 200, 300, 400, 500, 600, 610, 611, 612, 650, 700, 800, 900]
    mu_plotlist = [0, 400, 610, 800]


### Plot lattice for vsq:
figure(33)
errorbar(TD_lattice_mu400['T'], TD_lattice_mu400['cs2'], TD_lattice_mu400['dcs2'], ls='', marker='s', ms=MS, mew=MEW, mec='r', ecolor='r', color='r', alpha=0.5)




Tc_muT = np.zeros((2, len(mulist)))
k = 0
kk = 0
plot_p_loop = 1
print p_rasters_const_mu.keys()
#print buh
for mu in mulist:
    print mu
    p_list = p_rasters_const_mu[mu]
    TD_n = p_list[3]

    tdv_raster = p_list[2]
    p_raster = p_list[0]
    ps_raster = p_list[0]/p_list[2]**4.0

    print tdv_raster
    p_PTc = p_PT_calc(p_raster, ps_raster, tdv_raster)
    print 4*'#'
    print p_PTc
    print 4*'#'
    if p_PTc[0] == '1st':
        Tc_muT[:, kk] = np.array([mu, p_PTc[3]])
        inds = p_PTc[4]
        Tc = p_PTc[3]

    T_raster = p_list[2]
    if mu in mu_plotlist:
        ### p/T^4 ###
        figure(30)
        if plot_p_loop:
            plot(p_list[2], p_list[1], lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu = %d \, MeV$' %mu)
        else:
            if p_PTc[0] == 'ifl':
                plot(p_list[2], p_list[0]/p_list[2]**4.0, lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu = %d \, MeV$' %mu)
            else:
                plot(T_raster[:inds[0]], p_list[0][:inds[0]]/T_raster[:inds[0]]**4.0, lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu = %d \, MeV$' %mu)
                plot(T_raster[inds[1] - 1:], p_list[0][inds[1] - 1:]/T_raster[inds[1] - 1:]**4.0, lw = 2, color = clrs[
                    k], ls = lss[k])

        ### s/T^3 ###
        figure(31)
        #if p_PTc[0] == 'ifl':
        plot(p_list[2], TD_n[:, 1]/p_list[2]**3.0, lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu = %d \, MeV$' %mu)
        # else:
        #     plot(T_raster[:inds[0]], TD_n[:, 1][:inds[0]]/T_raster[:inds[0]]**3.0, lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu = 'r'%d \, 'r'MeV$' %mu)
        #     plot(T_raster[inds[1]:], TD_n[:, 1][inds[1]:]/T_raster[inds[1]:]**3.0, lw = 2, color = clrs[k], ls = lss[k])
        #if  p_PTc[0] != 'ifl':
            #axvline(x = Tc, color = clrs[k], ls = lss[k], lw = 1)


        ### I/T^4 ###
        I = e_and_I(TD_n, p_list[0])[1]
        figure(32)
        if p_PTc[0] == 'ifl':
            plot(p_list[2], I/p_list[2]**4.0, lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu = %d \, MeV$' %mu)
        else:
            plot(T_raster[:inds[0]], I[:inds[0]]/T_raster[:inds[0]]**4.0, lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu = %d \, MeV$' %mu)
            plot(T_raster[inds[1]:], I[inds[1]:]/T_raster[inds[1]:]**4.0, lw = 2, color = clrs[k], ls = lss[k])


        ### v_s^2 ###
        vsq_raster = vsq(TD_n, 'T')
        figure(33)
        if p_PTc[0] == 'ifl':
            plot(p_list[2], vsq_raster, lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu = %d \, MeV$' %mu)
        else:
            plot(T_raster[:inds[0]], vsq_raster[:inds[0]], lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu = %d \, MeV$' %mu)
            plot(T_raster[inds[1]:], vsq_raster[inds[1]:], lw = 2, color = clrs[k], ls = lss[k])
        if model_type=='no':
            plot(nocurves[400]['vsq'][:,0], nocurves[400]['vsq'][:,1], c='g', label='no paper')


        ### rho/T^3 ###
        figure(34)
        if p_PTc[0] == 'ifl':
            plot(p_list[2], TD_n[:,3]/p_list[2]**3.0, lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu = %d \, MeV$' %mu)
        else:
            plot(T_raster[:inds[0]], TD_n[:,3][:inds[0]]/T_raster[:inds[0]]**3.0, lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu = %d \, MeV$' %mu)
            plot(T_raster[inds[1]:], TD_n[:,3][inds[1]:]/T_raster[inds[1]:]**3.0, lw = 2, color = clrs[k], ls = lss[k])

        ### s/n ###
        figure(35)
        if mu > 0:
            if p_PTc[0] == 'ifl':
                plot(p_list[2], TD_n[:,1]/TD_n[:,3], lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu 'r'= %d \, MeV$' %mu)
            else:
                plot(T_raster[:inds[0]], TD_n[:,1][:inds[0]]/TD_n[:,3][:inds[0]], lw = 2, color = clrs[k], ls = lss[k], label = r'$\mu = %d \, MeV$' %mu)
                plot(T_raster[inds[1]:], TD_n[:,1][inds[1]:]/TD_n[:,3][inds[1]:], lw = 2, color = clrs[k], ls = lss[k])
        # if mu > 0:
        #     chi2_raster = chi2_calc(TD_n)
        #     figure(35)
        #     plot(p_list[2], chi2_raster/p_list[2]**2.0, color = clrs[k], label = r'$\mu = %d \, MeV$' %mu)
        k += 1
    kk += 1

#Tc_muT = np.compress(Tc_muT != 0, Tc_muT)
print 'Tc_muT:', Tc_muT

if plot_PDstuff:
    figure(20)
    mu_Tc = np.compress(Tc_muT[0,:] != 0, Tc_muT[0,:])
    T_Tc = np.compress(Tc_muT[1,:] != 0, Tc_muT[1,:])
    plot(mu_Tc, T_Tc, color = 'k', lw = 3, label = r'$T_c(\mu, T)$')
    legend(frameon = True, loc = 'upper right')



if model_type == 'no' or 'VRY_2' or 'VRY_4':
    figure(30) # pT4
    axis([100, 500, 0, 4.5])
    figure(31) # sT3
    axis([100, 500, 0, 18])
    ax = subplot(111)
    ax.set_yticks(np.arange(0, 19, 3))
    figure(32) # IT4
    axis([100, 500, 0, 18])
    figure(33) #vsq
    axis([100, 500, 0.0, 0.35])
    figure(34) # nT3
    axis([100, 500, 0, 2.6])
    ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    figure(35) # sn
    axis([100, 500, 0e3, 0.7e2])
    #ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    MLs = {30:[100, 20, 1, 0.2], 31:[100, 20, 5, 1], 32:[100, 20, 5, 1], 33:[100, 20, 0.1, 0.02], 34:[100, 20, 0.5, 0.1], 35:[100, 20, 50, 10]}

ylbls = {30:r'$p/T^{\,4}$', 31:r'$s/T^{\,3}$', 32:r'$I/T^{\,4}$', 33:r'$v_s^2$', 34:r'$n/T^{\,3}$', 35:r'$s/n$'}
savenames = {30:'pT4', 31:'sT3', 32:'IT4', 33:'vsq', 34:'nT3', 35:'sn'}

for fig in [30, 31, 32, 33, 34, 35]:
    figure(fig)
    xlabel(r'$T \, [MeV\,]$')
    ylabel(ylbls[fig])
    ax = subplot(111)
    ax.set_xticks(np.array([100, 200, 300, 400, 500]))
    nice_ticks()
    #if fig==31:
        #legend(loc = 'best', numpoints=3, fontsize = lfs, frameon = 0, fancybox = 0, columnspacing = 1)
    savefig(model_type+'/'+ftype+'/pdfs/curves/'+model_type+'_curves_'+savenames[fig]+'.pdf')

show()


#axlim1 = [0, 600.0, -0.3, 50.0]
#axlim2 = [0, 600.0, -7.0/150.0, 7.0]
#axlim3 = [0, 3000.0, -0.2/150.0, 0.2]
#axlim4 = [0, 3000.0, -0.02/150.0, 0.02]

#axlims = [axlim1, axlim2, axlim3, axlim4]
#ylbls = [r'$s/T^{\,3}$', r'$n/T^{\,3}$', r'$s/\mu^3$', r'$\rho/\mu^3$']
#lglocs = ['upper right', 'best', 'upper right', 'upper right']

#axlim1SBs = [0, 600.0, -0.01, 1.0]
#axlim2SBs = [0, 600.0, -0.01, 1.0]
#axlim3SBs = [0, 3000.0, -0.01, 1.0]
#axlim4SBs = [0, 3000.0, -0.01, 1.0]

#axlimsSBs = [axlim1SBs, axlim2SBs, axlim3SBs, axlim4SBs]
#ylblsSBs = [r'$s/s_{SB}$', r'$\rho/\rho_{SB}$', r'$s/s_{SB}$', r'$\rho/\rho_{SB}$']

#lglocsSBs = ['upper right', 'best', 'upper right', 'upper right']

#figure(1)
#xlabel(r'$T \,[MeV]$')

#figure(2)
#xlabel(r'$T \,[MeV]$')

#figure(3)
#xlabel(r'$\mu [MeV]$')

#figure(4)
#xlabel(r'$\mu [MeV]$')

#ffac = 1.0
#for i in range(1, 5):
    #figure(i, figsize = (ffac*0.8, ffac*0.6))
    #ax = subplot(111)
    ##ax = subplot(111)
    #axis(axlimsSBs[i - 1])
    #ylabel(ylblsSBs[i - 1])
    ##legend(fontsize = 16, labelspacing = 0.03, loc = lglocsSBs[i - 1])
    #ax.set_position(Bbox([[0.17, 0.14], [0.95, 0.95]]))
    #for l in ax.get_xticklines() + ax.get_yticklines():
        #l.set_markersize(8)
        #l.set_markeredgewidth(3)

    ##savefig('fig%d.pdf' %i)


#############################
#f, ((ax1, ax2), (ax3, ax4)) = subplots(2, 2, sharex='col', sharey='row')
#f.set_size_inches(12.0, 10.0, forward=True)
##ax = subplot(111)

#for i in range(0, len(T_levels)):
    #lvl = T_levels[i]
    #acc_cntrTD = tmc['T'][lvl]

    #TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]
    #ax2.semilogy(TD_slice_scaled[:, 2]*0.1, TD_slice_scaled[:, 1], color = cnt1_colors[i], label = r'$T = '+str(lvl)+'$ [MeV]')
    #ax4.semilogy(TD_slice_scaled[:, 2]*0.1, TD_slice_scaled[:, 3], color = cnt1_colors[i], label = r'$T = '+str(lvl)+'$ [MeV]')

#for i in range(0, len(mu_levels)):
    #lvl = mu_levels[i]
    #acc_cntrTD = tmc['mu'][lvl]

    #TD_slice_scaled = TD_scale_isen(acc_cntrTD[1], lambdas)[0]
    #ax1.semilogy(TD_slice_scaled[:, 0], TD_slice_scaled[:, 1], color = cnt2_colors[i], label = r'$\mu = '+str(lvl)+'$ [MeV]')
    #if lvl > 0:
        #ax3.semilogy(TD_slice_scaled[:, 0], TD_slice_scaled[:, 3], color = cnt2_colors[i])

#ax1.set_xlim([0, 300])
#ax2.set_xlim([0, 300])
#tight_layout()

#ax1.set_ylabel(r'$s [MeV]^3$')

#ax3.set_xlabel(r'$T \, [MeV]$')
#ax3.set_ylabel(r'$\rho [MeV]^3$')

#ax4.set_xlabel(r'$0.1 \mu [MeV]$')
#ax1.set_ylim([1e-1, 2*1e9])

##ax1.legend(loc = 'lower right', fontsize = 16, labelspacing = 0.3)
##ax2.legend(loc = 'lower center', fontsize = 16, ncol = 2, labelspacing = 0.3)

#tight_layout()
#subplots_adjust(left = 0.12, bottom = 0.1)

#axs = [ax1, ax2, ax3, ax4]
#for axi in axs:
    #for l in axi.get_xticklines() + axi.get_yticklines():
        #l.set_markersize(8)
        #l.set_markeredgewidth(2)

#show()
