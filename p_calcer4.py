import numpy as np
from scipy.integrate import quad, odeint
from scipy.interpolate import splrep, splev
from scipy.optimize import brentq

from pylab import figure, plot, legend, show, semilogy, scatter, xlabel, ylabel, rc, axis, savefig, subplot, contour, contourf, colorbar, cm, axhline, axvline, getp

def p_integrand(x, logp_integrand_tck, mpi_sgn):
    logp_integrand_tck = logp_integrand_tck[0]
    return mpi_sgn*(np.exp(splev(x, logp_integrand_tck)) - 1.0)

def add_TD_init_pt(TD, Phiphi, muorT_val, mu_or_T, l_ind, r_ind):
    T_raster = TD[:, 0]

    if T_raster[0] > T_raster[-1]:
        for k in range(0, 4):
            TD[:, k] = TD[:, k][::-1]
            if k < 2:
                Phiphi[k, :] = Phiphi[k, :][::-1]
        T_raster = TD[:, 0]

    mu_raster = TD[:, 2]
    if mu_or_T == 'T':
        x_raster = T_raster
        y_raster = mu_raster
        xind = 0
        yind = 2
    elif mu_or_T == 'mu':
        x_raster = mu_raster
        y_raster = T_raster
        xind = 2
        yind = 0

    val_ind = len(np.compress(x_raster < muorT_val, x_raster))

    #TD_new = np.zeros((val_ind, 4))
    #Phiphi_new = np.zeros((val_ind, 2))
    TD_new = np.zeros((len(TD[:, 0]) - val_ind + 1, 4))
    Phiphi_new = np.zeros((2, len(TD[:, 0]) - val_ind + 1))

    ind_l = np.amax([val_ind - 4, 0])
    ind_r = np.amin([val_ind + 4, len(T_raster)])
    print ind_l, val_ind, ind_r
    # print x_raster[ind_l:ind_r]
    #x_raster[ind_l:ind_r], y_raster[ind_l:ind_r] = zip(*sorted(zip(x_raster[ind_l:ind_r], y_raster[ind_l:ind_r])))
    y_val = splev(muorT_val, splrep(x_raster[ind_l:ind_r], y_raster[ind_l:ind_r]))
    TD_new[0, xind] = muorT_val
    TD_new[0, yind] = y_val
    # print 'yval:', y_val
    for i in [1, 3]:
        TDfunc_val = np.exp(splev(muorT_val, splrep(x_raster[ind_l:ind_r], np.log(TD[:, i][ind_l:ind_r]))))
        TD_new[0, i] = TDfunc_val

    for j in [0, 1]:
        Phiphi_val = splev(muorT_val, splrep(x_raster[ind_l:ind_r], Phiphi[j, ind_l:ind_r]))
        Phiphi_new[j, 0] = Phiphi_val

    # print Phiphi[0, :]

    for k in range(0, 4):
        TD_new[1:, k] = TD[val_ind:, k]
    for l in range(0, 2):
        Phiphi_new[l, 1:] = Phiphi[l, val_ind:]

    # print 'T =', TD_new[:, 0]
    # print 's =', TD_new[:, 1]
    # print 'mu =', TD_new[:, 2]
    # print 'rho =', TD_new[:, 3]
    # print Phiphi_new[0, :]
    print 'finished adding TD point'
    return [TD_new, Phiphi_new]

def p_calc_line(Tmu_i, Tmu_f, TD, Phiphi):
    T_i = Tmu_i[0]
    mu_i = Tmu_i[1]
    T_f = Tmu_f[0]
    mu_f = Tmu_f[1]

    T_raster = TD[:, 0]
    s_raster = TD[:, 1]
    mu_raster = TD[:, 2]
    rho_raster = TD[:, 3]

    phi0_raster = Phiphi[1, :] #check
    Phi1r_raster = Phiphi[0, :] #check

    print 'phi0 =', phi0_raster, 'Phi1r = ', Phi1r_raster
    print 'T =', T_raster, 'mu =', mu_raster

    ##parametrization:
    if mu_i == 0 and mu_f == 0:
        x_raster = phi0_raster
        TD_var = T_raster
        ind_init = -1
        TDf_raster = s_raster
        mpi_sgn = - 1.0
        intover = 'T, s:'
        print 'integrating on T-axis along T-axis'
    elif mu_i == 0 and mu_f > 0:
        #if T_f < 200:
        x_raster = np.arctan(Phi1r_raster/phi0_raster) ##angle between point on T=const curve and T axis
        x_raster = x_raster**(1.0/3.0)
        # else:
        #     x_raster = phi0_raster
        #x_raster = (Phi1r_raster/phi0_raster)**(1.0/4.0)
        TD_var = mu_raster
        ind_init = 0 #check
        TDf_raster = rho_raster
        mpi_sgn = + 1.0  ## "-" to have positive m_p_integrand
        intover = 'mu, rho:'
        print 'integrating from T-axis along mu-axis into the mu-plane'
    elif mu_i > 0 and mu_i == mu_f:
        x_raster = phi0_raster#[::-1]
        TD_var = T_raster
        ind_init = 0
        TDf_raster = s_raster
        mpi_sgn = - 1.0
        intover = 'T, s:'
        print 'integrating in the T-mu-plane along the T-axis'
        #plot(T_raster, s_raster)
        #show()

    print 'x_raster =', x_raster
    print 'TDf_raster =', TDf_raster

    if x_raster[0] > x_raster[-1]:
        dlogtdf_dx_tck = splrep(x_raster[::-1], np.log(1.0 + TDf_raster[::-1]))
    else:
        dlogtdf_dx_tck = splrep(x_raster, np.log(1.0 + TDf_raster))

    dtfdx = splev(x_raster, dlogtdf_dx_tck, der = 1)*(1.0 + TDf_raster)
    m_p_integrand = mpi_sgn*TD_var*dtfdx

    #if mu_i > 0 and mu_i == mu_f:
        #plot(x_raster, TDf_raster)
        #plot(x_raster, np.log(1.0 + TDf_raster))
        #plot(x_raster, splev(x_raster, dlogtdf_dx_tck, der = 1))
        #plot(x_raster, dtfdx)
        #plot(x_raster, m_p_integrand)
        #plot(x_raster, np.log(1.0 + m_p_integrand))
        #show()

    if x_raster[0] > x_raster[-1]:
        p_integrand_tck = splrep(x_raster[::-1], np.log(1.0 + m_p_integrand)[::-1])
    else:
        p_integrand_tck = splrep(x_raster, np.log(1.0 + m_p_integrand))

    p_raster = np.zeros(len(phi0_raster))
    x_0 = x_raster[ind_init]

    # xr2 = np.linspace(x_raster[0], x_raster[-1], 5000)
    # if mu_i > 0 and mu_i == mu_f:
    #     # figure(9)
    #     # plot(x_raster, TDf_raster)
    #     figure(10)
    #     plot(x_raster, np.log(1.0 + TDf_raster))
    #     plot(xr2, splev(xr2, dlogtdf_dx_tck))
    #
    #     # figure(12)
    #     # plot(x_raster, m_p_integrand)
    #
    #     figure(14)
    #     plot(x_raster, np.log(1.0 + m_p_integrand))
    #     plot(xr2, splev(xr2, p_integrand_tck))
    #     #show()

    print 'initial vals:', intover, TD_var[ind_init], TDf_raster[ind_init]
    for j in range(0, len(phi0_raster)):
        integral = quad(p_integrand, x_0, x_raster[j], args = ([p_integrand_tck],mpi_sgn), limit = 400, epsrel = 1e-10, epsabs = 1e-10)[0]
        p_raster[j] = TD_var[j]*TDf_raster[j] - TD_var[ind_init]*TDf_raster[ind_init] - integral

    return [p_raster, TD_var]

def get_p_init(p_init_list, muorT_val, inter_func):
    p_i_raster = p_init_list[0]
    muorT_i_raster = p_init_list[1]
    p_sc_i_raster = p_i_raster/muorT_i_raster**4.0
    #muorT_i_raster = p_init_list[2]
    #p_sc_i_raster = p_init_list[1]

    #muorT_i_raster, p_i_raster, p_sc_i_raster = zip(*sorted(zip(muorT_i_raster, p_i_raster, p_sc_i_raster )))
    if muorT_i_raster[0] > muorT_i_raster[-1]:
        p_i_raster = p_i_raster[::-1]
        p_sc_i_raster = p_sc_i_raster[::-1]
        muorT_i_raster = muorT_i_raster[::-1]

    print 'mu or T val:', muorT_val
    print 'mu or T init:', muorT_i_raster
    #print np.nan_to_num(muorT_i_raster)
    muorT_val_ind = len(np.compress(muorT_i_raster < muorT_val, muorT_i_raster))
    l_ind = np.amax([muorT_val_ind - 3, 0])
    r_ind = np.amin([muorT_val_ind + 3, len(muorT_i_raster)])
    print 'indices:', l_ind, r_ind, muorT_val_ind
    p_i_raster = p_i_raster[l_ind:r_ind]
    p_sc_i_raster = p_sc_i_raster[l_ind:r_ind]
    muorT_i_raster = muorT_i_raster[l_ind:r_ind]
    print 'mu or T init:', muorT_i_raster

    if muorT_val in muorT_i_raster:
        p_0 = p_i_raster[np.where(muorT_i_raster == muorT_val)[0][0]]
    else:
        if inter_func == 'p scaled':
            y_raster = p_sc_i_raster
        elif inter_func == 'p':
            y_raster = p_i_raster
        elif inter_func == 'log p':
            y_raster = np.log(p_i_raster)

        print 'interpolating '+inter_func+' to get value of p_0'
        print len(muorT_i_raster)
        if len(muorT_i_raster)==3:
            y_tck = splrep(muorT_i_raster, y_raster, k=2)
        else:
            y_tck = splrep(muorT_i_raster, y_raster)
        y_0 = splev(muorT_val, y_tck)

        if inter_func == 'p scaled':
            p_0 = y_0*muorT_val**4.0
        elif inter_func == 'p':
            p_0 = y_0
        elif inter_func == 'log p':
            p_0 = np.exp(y_0)

    print 'for ', muorT_val, 'p_0 =', p_0, 'p_0 scaled =', p_0/muorT_val**4.0
    return [p_0, l_ind, r_ind]

def p_calc_Tlvl(p_on_T_axis, TD_T_const, Phiphi_T_const, T_val):
    mu_raster = TD_T_const[:, 2]
    mu_f = mu_raster[-1]
    p_0 =  get_p_init(p_on_T_axis, T_val, 'p scaled')[0] # splev(T_val, splrep(p_on_T_axis[1][::-1], p_on_T_axis[0][::-1]))
    pcl = p_calc_line([T_val, 0], [T_val, mu_f], TD_T_const, Phiphi_T_const)
    p_raster_Tlvl = p_0 + pcl[0]

    return [p_raster_Tlvl, p_raster_Tlvl/pcl[1]**4.0, pcl[1]]

def p_calc_mulvl(p_raster_Tlvl, TD_mu_const, Phiphi_mu_const, mu_val, T_i, T_f):
    T_raster = TD_mu_const[:, 0]
    print 'T_raster, mu_const: ', TD_mu_const[:, 0], TD_mu_const[:, 2]
    # T_i = T_raster[0]
    # T_f = T_raster[-1]
    gpi = get_p_init(p_raster_Tlvl, mu_val, 'p scaled')

    p_0 = gpi[0]
    l_ind = gpi[1]
    r_ind = gpi[2]
    TDPP = add_TD_init_pt(TD_mu_const, Phiphi_mu_const, T_i, 'T', l_ind, r_ind)
    TD_mu_const = TDPP[0]
    Phiphi_mu_const = TDPP[1]

    print 'TD_mu_const: ', TD_mu_const[:, 0], TD_mu_const[:, 2]
    pcl = p_calc_line([T_i, mu_val], [T_f, mu_val], TD_mu_const, Phiphi_mu_const)
    p_raster_mulvl = p_0 + pcl[0]

    return [p_raster_mulvl, p_raster_mulvl/pcl[1]**4.0, pcl[1], TD_mu_const, Phiphi_mu_const]

#def p_calc_Taxis(TD_T_axis, phi0_raster):
#    T_raster = TD_T_axis[:, 0]
#    s_raster = TD_T_axis[:, 1]
#    w_raster = T_raster*s_raster
#    p_raster = np.zeros(len(phi0_raster))
#
#    dsdphi0 = splev(phi0_raster, splrep(phi0_raster, np.log(s_raster)), der = 1)*s_raster
#    mp_integrand_raster = - T_raster*dsdphi0
#    logmp_integrand_tck = splrep(phi0_raster, np.log(mp_integrand_raster))
#
#    phih_inf = np.amax(phi0_raster)
#    for j in range(0, len(p_raster)):
#        p_raster[j] = w_raster[j] - w_raster[-1] + quad(p_integrand, phih_inf, phi0_raster[j], args = ([logmp_integrand_tck]), limit = 400, epsrel = 1e-10, epsabs = 1e-10)[0]
#
#    return [p_raster, p_raster/T_raster**4.0, T_raster]

def tdv_PT_finder(tdv, ltp_tck, htp_tck):
    return splev(tdv, htp_tck) - splev(tdv, ltp_tck)

def p_PT_calc(p_raster, ps_raster, tdv_raster):
    if tdv_raster[-1] < tdv_raster[0]:
        tdv_raster = tdv_raster[::-1]
        p_raster = p_raster[::-1]
        ps_raster = ps_raster[::-1]

    print 'tdv_raster: ', tdv_raster
    #plot(tdv_raster, p_raster, ls='', marker='s')
    #show()

    PT_type = 'ifl'
    for i in range(1, len(tdv_raster)):
        if tdv_raster[i] < tdv_raster[i - 1]:
            PT_type = '1st'
            ltp_maxind = i - 1
            break
    print 'phase transition type:', PT_type

    if PT_type == '1st':
        for i in range(1, len(tdv_raster))[::-1]:
            if tdv_raster[i] < tdv_raster[i - 1]:
                htp_minind = i
                break

        print 'ltp_maxind =', ltp_maxind, 'htp_minind =', htp_minind
        ## get low temperature and high temperaure phases
        p_ltp = p_raster[:ltp_maxind + 1]
        ps_ltp = ps_raster[:ltp_maxind + 1]
        tdv_ltp = tdv_raster[:ltp_maxind + 1]

        p_htp = p_raster[htp_minind:]
        ps_htp = ps_raster[htp_minind:]
        tdv_htp = tdv_raster[htp_minind:]

        print 'tdv_ltp =', tdv_ltp, 'tdv_htp =', tdv_htp, 'tdv_up =', tdv_raster[ltp_maxind + 1:htp_minind]
        ## cut-off low temperature and high temperature phases using mu or T spinodals
        ltp_minind = np.amax([0, len(np.compress(tdv_ltp < tdv_htp[0], tdv_ltp)) - 3])
        htp_maxind = np.amin([len(tdv_htp), len(np.compress(tdv_htp < tdv_ltp[-1], tdv_htp)) + 2])

        p_ltpc = p_ltp[ltp_minind:]
        ps_ltpc = ps_ltp[ltp_minind:]
        tdv_ltpc = tdv_ltp[ltp_minind:]

        p_htpc = p_htp[:htp_maxind + 1]
        ps_htpc = ps_htp[:htp_maxind + 1]
        tdv_htpc = tdv_htp[:htp_maxind + 1]

        ## compute spline representations of low-temp and high-temp curves p_scaled(mu or T)
        ps_ltp_tck = splrep(tdv_ltpc, ps_ltpc)
        ps_htp_tck = splrep(tdv_htpc, ps_htpc)
        p_ltp_tck = splrep(tdv_ltpc, p_ltpc)
        p_htp_tck = splrep(tdv_htpc, p_htpc)
        #plot(tdv_ltpc, p_ltpc)
        #plot(tdv_htpc, p_htpc)


        ## a and b vals for brentq = borders of the spinodals
        tdv_l = tdv_htp[0]
        tdv_r = tdv_ltp[-1]
        print 'tdv_ltpc, tdv_htpc: ', tdv_ltpc, tdv_htpc

        ## compute pressure and T or mu of 1st order phase transition
        #tdv_PT = brentq(tdv_PT_finder, tdv_l, tdv_r, xtol=1e-12, rtol=1e-10, args = (ps_ltp_tck, ps_htp_tck))#[0]
        tdv_PT = brentq(tdv_PT_finder, tdv_l, tdv_r, xtol=1e-12, rtol=1e-10, args = (p_ltp_tck, p_htp_tck))
        ps_PT = np.float(splev(tdv_PT, ps_htp_tck))
        p_PT = ps_PT*tdv_PT**4.0

        print 'TD val PT:', tdv_PT, 'p scaled PT:', ps_PT, 'p PT:', p_PT
        #show()

        ## indices for original p and Tormu arrays for true high-T and low-T phases
        ltp_ind = len(np.compress(tdv_ltp <= tdv_PT, tdv_ltp))
        htp_ind = len(tdv_raster) - len(np.compress(tdv_htp >= tdv_PT, tdv_htp))
        up_ind = len(tdv_ltp) + len(np.compress(tdv_raster[ltp_maxind + 1:htp_minind] >= tdv_PT, tdv_raster[ltp_maxind + 1:htp_minind])) - 1

        print 'ltp_ind =', ltp_ind, 'htp_ind =', htp_ind, 'up_ind =', up_ind, 'tdvs:', tdv_raster[ltp_ind], tdv_raster[htp_ind], tdv_raster[up_ind]

        return [PT_type, p_PT, ps_PT, tdv_PT, [ltp_ind, htp_ind, up_ind], [ltp_maxind, htp_minind], [ltp_minind, htp_maxind]]
    else:
        return [PT_type]