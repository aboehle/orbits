import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm,colors
import matplotlib.gridspec as gridspec
import scipy
from astropy.io import ascii
import h5py
import glob
from plotting.usetextrue import *

import orbits.kepler as k
from plotting.plot_contrastcurve import calc_limitingmass

# rootDir = '/data/ipa/meyer/boehlea/data/'
rootDir = '/Volumes/ipa/quanz/user_accounts/aboehle/data/'

dataDirs = {'tau_ceti': 'tau_ceti_lprime/data_with_raw_calibs_newpipeline/',
            '40_eri': '40eri_lprime_081116/data_with_raw_calibs/',
            'axmic': 'axmic_lprime_091011/data_with_raw_calibs/',
            'kapteyns': 'kapteyns_lprime_081116/data_with_raw_calibs_fmenti/',
            'hd36395': 'hd36395_lprime_081116/data_with_raw_calibs_fmenti/',
            'hd42581': 'hd42581_lprime_081116/data_with_raw_calibs/'}


# idea: read in the full calibrated RV time series, calculate standard deviation from that in the code below!
def get_rv_timeseries(star):
    """
    Find the times in JD of the RV observations for the inputed star.

    :param star: name of star
    :type star: str

    :return: tuple with array of epochs of RV observations in JD and array of calibrated RV measurements
    :rtype: (ndarray,ndarray)
    """

    rvDir = '/Users/annaboehle/research/analysis/archival_stars/rv_calibrated'

    file_ls = glob.glob('{:s}/{:s}/{:s}*cal.dat'.format(rvDir, star, star))
    rv_epochs = []
    rv_meas = []
    for f in file_ls:
        rv_tab = ascii.read(f,names=('rjd','vrad'))
        # rv_epochs.append(ascii.read(f))
        rv_epochs.extend(rv_tab['rjd'][:])
        rv_meas.extend(rv_tab['vrad'][:])

    print 'Number of RV epochs:', len(np.array(rv_epochs, dtype=float))

    return np.array(rv_epochs, dtype=float) + 2400000, np.array(rv_meas)


def get_rv_epochs(star):
    """
    Find the times in JD of the RV observations for the inputed star.

    :param star: name of star
    :type star: str

    :return: array of epochs of RV observations in JD
    :rtype: ndarray
    """
    
    rvDir = '/Users/annaboehle/research/data/rv/dace/nightly_binning'

    file_ls = glob.glob('{:s}/{:s}/{:s}*Residuals.rdb'.format(rvDir,star,star))
    rv_epochs = []
    for f in file_ls:
        rv_tab = ascii.read(f)
        # rv_epochs.append(ascii.read(f))
        rv_epochs.extend(rv_tab['rjd'][1:])

    print 'Number of RV epochs:',len(np.array(rv_epochs,dtype=float))
        
    return np.array(rv_epochs,dtype=float) + 2400000


def get_f_masslimvsep(star, d_star, age_star, mag_star, stack, n):
    """
    Make a function that maps the projected separation to the mass limit for a given star and its parameters,
    and for a given set of PSF subtraction parameters.

    :param star: name of star
    :type star: str
    :param d_star: distance of star in pc
    :type d_star: float
    :param age_star: age of star in Gyr
    :type age_star: float
    :param mag_star: apparent magnitude of star in the L' band
    :type mag_star: float
    :param stack: number of frames stacked together before PSF subtraction
    :type stack: int
    :param n: number of PCA components used to model PSF
    :type n: int

    :return: function mapping projected separation to mass limit, measured separations, measured mass limits
    :rtype: function, ndarray, ndarray
    """

    # get contrast limits versus projected sep for star
    f = h5py.File(rootDir + dataDirs[star] + 'PynPoint_database.hdf5','r')

    contrast_tag_inner = "{:s}_lprime_contrast_limits_stack{:02d}_pca{:02d}_0.1delr".format(star, stack, n)
    if star != '40_eri':
        if star in ['tau_ceti','hd42581']:
            stack = 10
        contrast_tag_outer = "{:s}_lprime_contrast_limits_stack{:02d}_adi".format(star, stack)            
        seps = np.concatenate((f[contrast_tag_inner][:,0],f[contrast_tag_outer][:,0]))
        contr = np.concatenate((f[contrast_tag_inner][:,1],f[contrast_tag_outer][:,1]))
    else:
        seps = f[contrast_tag_inner][:,0]
        contr = f[contrast_tag_inner][:,1]
    
    f.close()

    # convert contrast limits to mass limits
    mass_lims = np.zeros(len(contr))
    for cc in range(len(contr)):
        mass_lims[cc],t_model,below,above,t_eff = calc_limitingmass(d_star, age_star, mag_star, contr[cc], filt="L'", interp_age=True)
    
    f_masslimvsep = scipy.interpolate.interp1d(seps,mass_lims,bounds_error=False,fill_value=np.inf)  # outside bounds planets cannot be detected!

    return f_masslimvsep, seps, mass_lims


def sample_orbits(star,
                  age='nominal',
                  nsamples=10000,
                  test=True,
                  plot_text=False,
                  fig_ax_ls=[],
                  star_info = {},
                  t_hci = None,
                  f_masslimvsep = None,
                  seps = None,
                  calc_rv_std=True,
                  save=True):
    """
    Combine the HCI and RV constraints and find the percentage of planets detected with each method
    using a Monte Carlo analysis.

    :param star: name of star
    :type star: str
    :param age: age of the star to use for determining the mass limit (either 'nominal','min', or 'max')
    :type age: str
    :param nsamples: number of sample to perform for the Monte Carlo
    :type nsamples: int
    :param test: if True, only the hard-coded mass and semi-major axis are tested and

                 figures are plotted showing the samples for various combination of orbital parameters
    :type test: bool
    :param plot_text: if True, then include text on the plot to indicate the % of detected orbits in each mass/a bin
    :type plot_text: bool
    :param fig_ax_ls: list of figure and axes to plot the results on [fig, ax1, ax2, ax3, ax4]
    :type fig_ax_ls: list

    """

    # read in info tables
    if not star_info:
        star_info = ascii.read('/Users/annaboehle/research/code/directimaging_code/plotting/nearest_solar_dist_age.txt')
        row = np.where(star_info['object_name'] == star)[0][0]
    else:
        row = 0

    # get star info
    print star
    
    d_star = star_info['dist'][row]
    if age == 'nominal':
        age_star = star_info['age'][row]
    elif age == 'min':
        age_star = star_info['age_min'][row]
    elif age == 'max':
        age_star = star_info['age_max'][row]
    else:
        raise ValueError('age should be "nominal", "min", or "max"!')
    
    mag_star = star_info['mag_star'][row]
    m_star = star_info['m_star'][row]
    #rv_std = star_info['rv_std'][row]

    if calc_rv_std:
        t_rv, rv_meas = get_rv_timeseries(star)
        rv_std = np.std(rv_meas,ddof=1)
    else:
        t_rv = get_rv_epochs(star)
        rv_std = star_info['rv_std'][row]
    print '{:s}: rv_std = {:1.2f} m/s'.format(star,rv_std)

    if not t_hci:
        obsInfo = ascii.read('/Volumes/ipa/quanz/user_accounts/aboehle/my_papers/archival_stars/tables/NACO_obs.dat')
        row_obs = np.where(obsInfo['name'] == star)[0][0]
        t_hci = obsInfo['time'][row_obs]

    if not f_masslimvsep:
        obsInfo = ascii.read('/Volumes/ipa/quanz/user_accounts/aboehle/my_papers/archival_stars/tables/NACO_obs.dat')
        row_obs = np.where(obsInfo['name'] == star)[0][0]

        contrInfo = ascii.read('/Volumes/ipa/quanz/user_accounts/aboehle/my_papers/archival_stars/tables/contr_info.dat')
        row_contr = np.where(contrInfo['name'] == star)[0][0]

        stack, p = contrInfo['stack'][row_contr], contrInfo['percen_frames'][row_contr]
        n = int(np.round(p * obsInfo['nframes_used'][row_obs] / stack))

        # get mass limits v projected sep for star
        f_masslimvsep, seps, mass_lims = get_f_masslimvsep(star, d_star, age_star, mag_star, stack, n)
    
    # fix orbital parameters to 0
    e, o, t0 = 0., 0., 0.

    # sample i and w
    unif = np.random.uniform(size=nsamples)
    i = np.arccos(unif)*(180./np.pi)
    #i = np.zeros(nsamples) + 20.

    w = np.random.uniform(size=nsamples, low=0, high=360.)

    # for a and m_p
    del_m_p = 3.0
    del_a = 2.
    if test:
        a_arr = [22]
        m_p_arr = [15]
    else:
        #m_p_arr = np.arange(np.ma.min(np.ma.masked_invalid(mass_lims))-5.0, 50, del_m_p)
        m_p_arr = np.arange(del_m_p/2., 50, del_m_p)
        
        #a_arr = np.arange(np.min(seps)*d_star, np.max(seps)*d_star+18.0,del_a)
        #a_arr = np.arange(np.min(seps)*d_star, np.max(seps)*d_star*4.0,del_a)
        if star == '40_eri':
            a_arr = np.arange(del_a/2., np.max(seps)*d_star*4.0,del_a)
        else:
            a_arr = np.arange(del_a/2., np.max(seps)*d_star*2.0,del_a)
    print m_p_arr,'M_J'
    print a_arr,'AU'

    # set up plot
    usetexTrue()

    if not fig_ax_ls:
        fig = plt.figure(figsize=(12,18))  #(12,14)
        gs = gridspec.GridSpec(4, 1)
        gs_cbar = gridspec.GridSpec(4,1) 
        gs.update(left=0.1,top=0.97,bottom=0.075,hspace=0.25,right=0.83,wspace=0.3)
        gs_cbar.update(top=0.97,bottom=0.075,left=0.86,right=0.88,hspace=0.25)
    
        ax1=plt.subplot(gs[0])    
        ax2=plt.subplot(gs[1])
        ax3=plt.subplot(gs[2])
        ax4=plt.subplot(gs[3])
    else:
        fig,ax1,ax2,ax3,ax4 = fig_ax_ls

    norm = colors.Normalize(vmin=0,vmax=100)
    scalar_map = cm.ScalarMappable(norm=norm,cmap=cm.Blues)
        
    norm = colors.Normalize(vmin=0,vmax=100)
    scalar_map_diff = cm.ScalarMappable(norm=norm,cmap=cm.Greens)

    dtype = np.dtype([('a','float'),
                      ('m_p','float'),
                      ('frac_detected_hci','float'),
                      ('frac_detected_rv','float'),
                      ('frac_detected_rvorhci','float'),
                      ('frac_rvhcidiff','float'),                      
                      ])
    
    #out_arr = np.zeros((len(m_p_arr)*len(a_arr), ))  # cols: a, m_p, frac_hci, frac_rv, frac_hciorrv, frac_hcinotrv
    out_arr = []
    print type(m_star)
    
    for a in a_arr:
        for m_p in m_p_arr:
            #a = 15. # AU
            #m_p = 18. # M_J
        
            # convert a to p
            P = np.sqrt(a**3./(m_star+m_p*954.7919e-6))*365.

            # solve kepler's equation
            E_rv = k.solve_kepler_eqn(t_rv, e, P, t0)
            E_hci = k.solve_kepler_eqn(t_hci, e, P, t0)
    
            # get rv prediction
            x_rv, y_rv, rv_rv = k.orbit_solution(E_rv,
                                                 e=e, P=P, t0=t0, i=i, w=w, o=o, m_p=m_p, m_star=m_star,d_star=d_star,
                                                 units = 'arcsec',
                                                 solve_kepler=False)
    
            # find astrometric location for these parameters
            x_ast, y_ast, rv_ast = k.orbit_solution(E_hci,
                                                    e=e, P=P, t0=t0, i=i, w=w, o=o, m_p=m_p, m_star=m_star,d_star=d_star,
                                                    units = 'arcsec',
                                                    solve_kepler=False)

            # get frac detected for imaging
            proj_sep = (x_ast.flatten()**2. + y_ast.flatten()**2.)**0.5
            mass_lims = f_masslimvsep(proj_sep)

            idx_detected_hci = np.where(m_p > mass_lims)[0]
            frac_detected_hci = len(idx_detected_hci)/float(nsamples)

            # get frac detected for RV
            rv_diff = np.max(rv_rv,axis=0) - np.min(rv_rv,axis=0)

            idx_detected_rv = np.where(rv_diff > 5*rv_std)[0]
            idx_notdetected_rv = np.where(rv_diff <= 5*rv_std)[0]
            frac_detected_rv = len(idx_detected_rv)/float(nsamples)

            #if test:
            #    if a == a_arr[10] and m_p == m_p_arr[0]:
            #        plt.figure()
            #        plt.hist(rv_diff)
            #        plt.figure()
            #        plt.errorbar(t_rv,rv_rv[:,idx_notdetected_rv[10]],marker='o',yerr=rv_std)

            # get frac detected either or
            idx_detected_rvorhci = np.where( (rv_diff > 5*rv_std) | (m_p > mass_lims) )[0]
            frac_detected_rvorhci = len(idx_detected_rvorhci)/float(nsamples)

            # plot imaging
            rect = Rectangle( (a-del_a/2.,m_p-del_m_p/2.),
                              del_a,del_m_p,
                              facecolor=scalar_map.to_rgba(frac_detected_hci*100.),edgecolor='black')
            ax1.add_patch(rect)
            ax1.scatter(a,m_p)

            # plot RV
            rect = Rectangle( (a-del_a/2.,m_p-del_m_p/2.),
                                del_a,del_m_p,
                                facecolor=scalar_map.to_rgba(frac_detected_rv*100.),edgecolor='black')
            ax2.add_patch(rect)
            ax2.scatter(a,m_p)


            # plot both!
            rect = Rectangle( (a-del_a/2.,m_p-del_m_p/2.),
                              del_a,del_m_p,
                              facecolor=scalar_map.to_rgba(frac_detected_rvorhci*100.),edgecolor='black')
            ax3.add_patch(rect)
            ax3.scatter(a,m_p)

            # plot difference!
            rect = Rectangle( (a-del_a/2.,m_p-del_m_p/2.),
                              del_a,del_m_p,
                              facecolor=scalar_map_diff.to_rgba((frac_detected_rvorhci-frac_detected_rv)*100.),edgecolor='black')
            ax4.add_patch(rect)
            ax4.scatter(a,m_p)

            if test:
                if (frac_detected_rvorhci-frac_detected_rv)*100. > 3:
                    idx_detected_hci_notrv =np.where( (rv_diff < 5*rv_std) & (m_p > mass_lims) )[0]
                    print len(idx_detected_hci_notrv)
                    print

                    idx_sort = np.argsort(t_rv)
                    turning_pt_idx = []
                    no_turning_idx = []                    
                    for idx in idx_detected_hci_notrv:
                        if nsamples <= 100:
                            plt.figure(2)
                            plt.plot(t_rv,rv_rv[:,idx]-rv_rv[:,idx][0],marker='.',linestyle='none')#,yerr=rv_std)
                                             
                        if not np.isclose(rv_diff[idx],np.abs(rv_rv[:,idx][idx_sort][0] - rv_rv[:,idx][idx_sort][-1])):
                            turning_pt_idx.append(idx)
                        else:
                            no_turning_idx.append(idx)

                    turning_pt_allidx = []
                    no_turning_allidx = []
                    for idx in range(nsamples):
                                             
                        if not np.isclose(rv_diff[idx],np.abs(rv_rv[:,idx][idx_sort][0] - rv_rv[:,idx][idx_sort][-1])):
                            turning_pt_allidx.append(idx)
                        else:
                            no_turning_allidx.append(idx)
                            
                    turning_pt_idx = np.array(turning_pt_idx,dtype=int)
                    no_turning_idx = np.array(no_turning_idx,dtype=int)                    
                    turning_pt_allidx = np.array(turning_pt_allidx,dtype=int)
                    no_turning_allidx = np.array(no_turning_allidx,dtype=int)                    
                            
                    print '% of RV curves with turning pt = {:2.2f}'.format(len(turning_pt_idx)/float(len(idx_detected_hci_notrv))*100.)

                    print a/d_star, '= max sep'
                    #print proj_sep[idx_detected_hci_notrv]
                    #print i[idx_detected_hci_notrv]

                    plt.figure(figsize=(10,8))
                    plt.axhline(a/d_star,color='black',label='max proj sep')
                    plt.plot(i,proj_sep,linestyle='none',marker='o',label='all pts')
                    plt.plot(i[turning_pt_allidx],proj_sep[turning_pt_allidx],linestyle='none',marker='o',label='yes turning in RV (all pts)')
                    plt.plot(i[turning_pt_idx],proj_sep[turning_pt_idx],linestyle='none',marker='o',label='yes turning in RV')
                    plt.plot(i[no_turning_idx],proj_sep[no_turning_idx],linestyle='none',marker='o',label='no turning in RV')
                    plt.xlabel('inclination (deg)')
                    plt.ylabel('projected sep (arcsec)')
                    plt.ylim(5,6.1)
                    #plt.xlim(0,60)
                    plt.legend()


                    plt.figure(figsize=(10,8))
                    plt.plot(i,w,linestyle='none',marker='o',label='all pts')
                    plt.plot(i[turning_pt_allidx],w[turning_pt_allidx],linestyle='none',marker='o',label='yes turning in RV (all pts)')               
                    plt.plot(i[turning_pt_idx],w[turning_pt_idx],linestyle='none',marker='o',label='yes turning in RV')
                    plt.plot(i[no_turning_idx],w[no_turning_idx],linestyle='none',marker='o',label='no turning in RV')
                    plt.xlabel('inclination (deg)')
                    plt.ylabel('w (deg)')
                    plt.legend()

                    plt.figure(figsize=(10,8))
                    plt.plot(rv_diff,proj_sep,linestyle='none',marker='o',label='all pts')
                    plt.plot(rv_diff[turning_pt_allidx],proj_sep[turning_pt_allidx],linestyle='none',marker='o',label='yes turning in RV (all pts)')                    
                    plt.plot(rv_diff[turning_pt_idx],proj_sep[turning_pt_idx],linestyle='none',marker='o',label='yes turning in RV')
                    plt.plot(rv_diff[no_turning_idx],proj_sep[no_turning_idx],linestyle='none',marker='o',label='no turning in RV')
                    plt.axhline(a/d_star,color='black',label='max proj sep')                    
                    plt.xlabel('rv diff (m/s)')
                    plt.ylabel('projected sep (arcsec)')
                    plt.ylim(5,6.1)
                    #plt.xlim(0,20)
                    plt.legend()

                    plt.figure(figsize=(10,8))
                    plt.plot(rv_diff,i,linestyle='none',marker='o',label='all pts')
                    plt.plot(rv_diff[turning_pt_allidx],i[turning_pt_allidx],linestyle='none',marker='o',label='yes turning in RV (all pts)')
                    plt.plot(rv_diff[turning_pt_idx],i[turning_pt_idx],linestyle='none',marker='o',label='yes turning in RV')
                    plt.plot(rv_diff[no_turning_idx],i[no_turning_idx],linestyle='none',marker='o',label='no turning in RV')
                    plt.xlabel('rv diff (m/s)')
                    plt.ylabel('inclination (deg)')
                    plt.xlim(0,12)
                    plt.ylim(0,60)
                    plt.legend()
                    

                    plt.figure()
                    
                    
                    
            # add text
            fontsize = 12
            if plot_text:                
                
                if (frac_detected_hci) > 0.005:
                    #if frac_detected_hci*100. > 70:
                    #    text_color = 'white'
                    #else:
                    text_color = 'black'
                    ax1.text(a, m_p, '{:2.0f}'.format(frac_detected_hci*100.),
                        verticalalignment='center',horizontalalignment='center',
                        color = text_color, fontsize=fontsize)

                if (frac_detected_rv) > 0.005:                    
                    #if frac_detected_rv*100. > 70:
                    #    text_color = 'white'
                    #else:
                    text_color = 'black'
                    ax2.text(a, m_p, '{:2.0f}'.format(frac_detected_rv*100.),
                        verticalalignment='center',horizontalalignment='center',
                        color = text_color, fontsize=fontsize)

                if (frac_detected_rvorhci) > 0.005:                    
                    #if frac_detected_rvorhci*100. > 70:
                    #    text_color = 'white'
                    #else:
                    text_color = 'black'
                    ax3.text(a, m_p, '{:2.0f}'.format(frac_detected_rvorhci*100.),
                        verticalalignment='center',horizontalalignment='center',
                        color = text_color, fontsize=fontsize)
                
            #if ((frac_detected_rvorhci-frac_detected_rv)*100.) > 70:
            #    text_color = 'white'
            #else:
                text_color = 'black'
                if (frac_detected_rvorhci-frac_detected_rv) > 0.005:
                    ax4.text(a, m_p, '{:2.0f}'.format((frac_detected_rvorhci-frac_detected_rv)*100.),
                            verticalalignment='center',horizontalalignment='center',
                            color = text_color, fontsize=fontsize)


            out_arr.append((a,
                            m_p,
                            frac_detected_hci,
                            frac_detected_rv,
                            frac_detected_rvorhci,
                            (frac_detected_rvorhci-frac_detected_rv),
                            ))

            # ultimately: frac detected by one method OR the other

    for ax in [ax1,ax2,ax3,ax4]:
        if ax:
            ax.set_xlim(0,a_arr[-1]+del_a/2.)
            ax.set_ylim(0,m_p_arr[-1]+del_m_p/2.)
        
    if not fig_ax_ls:
        ax_cbar1 = fig.add_subplot(gs_cbar[1])
        ax_cbar2 = fig.add_subplot(gs_cbar[3])
    
        scalar_map.set_array([])
        scalar_map_diff.set_array([])
        
        fig.colorbar(scalar_map,cax=ax_cbar1,label='\% of companions detected')
        fig.colorbar(scalar_map_diff,cax=ax_cbar2,label='Difference between\nHCI/RV combined and RV alone')

        for ax in [ax1,ax2,ax3,ax4]:
            ax.set_ylabel('')
        ax4.set_xlabel('Companion semi-major axis (AU)')
        fig.text(0.05,0.5,'Companion mass (M$_J$)',rotation=90,horizontalalignment='center',verticalalignment='center')

    if save:
        # write out results
        out_arr = np.array(out_arr,dtype=dtype)
        ascii.write(out_arr, 'hci_rv_lims/{:s}_constraints_{:s}age.dat'.format(star,age),overwrite=True,delimiter='\t')
    
    plt.show()
    
    return out_arr, scalar_map, scalar_map_diff

