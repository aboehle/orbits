import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm,colors
import scipy
from astropy.io import ascii
import h5py
import glob

import orbits.kepler as k
from plotting.plot_contrastcurve import calc_limitingmass

#rootDir = '/data/ipa/meyer/boehlea/data/'
rootDir = '/Volumes/ipa/meyer/boehlea/data/'

dataDirs = {'tau_ceti': 'tau_ceti_lprime/data_with_raw_calibs_newpipeline/',
            '40_eri': '40eri_lprime_081116/data_with_raw_calibs/',
            'axmic': 'axmic_lprime_091011/data_with_raw_calibs/',
            'kapteyns': 'kapteyns_lprime_081116/data_with_raw_calibs_fmenti/',
            'hd36395':'hd36395_lprime_081116/data_with_raw_calibs_fmenti/',
            'hd42581':'hd42581_lprime_081116/data_with_raw_calibs/'}

def get_rv_epochs(star):
    
    rvDir = '/Users/annaboehle/research/data/rv/'

    file_ls = glob.glob('{:s}{:s}/{:s}_resid*.rdb'.format(rvDir,star,star))
    rv_epochs = []
    for f in file_ls:
        rv_tab = ascii.read(f)
        #rv_epochs.append(ascii.read(f))
        rv_epochs.extend(rv_tab['rjd'][1:])

    return np.array(rv_epochs,dtype=float) + 2400000 
    
def get_f_masslimvsep(star, d_star, age_star, mag_star, stack, n):

    # get contrast limits versus projected sep for star
    f = h5py.File(rootDir + dataDirs[star] + 'PynPoint_database.hdf5','r')
    
    contrast_tag_inner = "{:s}_lprime_contrast_limits_stack{:02d}_pca{:02d}_0.1delr".format(star, stack, n)
    contrast_tag_outer = "{:s}_lprime_contrast_limits_stack{:02d}_adi".format(star, stack)
    seps = np.concatenate((f[contrast_tag_inner][:,0],f[contrast_tag_outer][:,0]))
    contr = np.concatenate((f[contrast_tag_inner][:,1],f[contrast_tag_outer][:,1]))
    
    f.close()

    # convert contrast limits to mass limits
    mass_lims = np.zeros(len(contr))
    for cc in range(len(contr)):
        mass_lims[cc],t_model,below,above,t_eff   = calc_limitingmass(d_star, age_star, mag_star, contr[cc], filt="L'", interp_age=True)        
    
    f_masslimvsep = scipy.interpolate.interp1d(seps,mass_lims,bounds_error=False,fill_value=np.inf)  # outside bounds planets cannot be detected!

    return f_masslimvsep, seps
    
def sample_orbits(test=True):

    # read in info tables
    star_info = ascii.read('/Users/annaboehle/research/code/directimaging_code/plotting/nearest_solar_dist_age.txt')
    contrInfo = ascii.read('/Volumes/ipa/meyer/boehlea/my_papers/archival_stars/tables/contr_info.dat')
    obsInfo = ascii.read('/Volumes/ipa/meyer/boehlea/my_papers/archival_stars/tables/NACO_obs.dat')
    
    # get star info
    star = 'hd36395'
    print star

    row_contr = np.where(contrInfo['name'] == star)[0][0]
    
    stack, p = contrInfo['stack'][row_contr],contrInfo['percen_frames'][row_contr]
    cent_size = contrInfo['cent_size'][row_contr]
    
    row = np.where(star_info['object_name'] == contrInfo['plt_contr_name'][row_contr])[0][0]
    
    d_star = star_info['dist'][row]
    age_star = star_info['age'][row]
    mag_star = star_info['mag_star'][row]
    t_baseline = star_info['t_baseline'][row]
    m_star = star_info['m_star'][row]
    rv_std = star_info['rv_std'][row]

    row_obs = np.where(obsInfo['name'] == star)[0][0]
    
    t_rv = get_rv_epochs(star)
    t_hci = obsInfo['time'][row_obs]
    n = int(np.round(p*obsInfo['nframes'][row_obs]/stack))

    # get mass limits v projected sep for star
    f_masslimvsep, seps = get_f_masslimvsep(star, d_star, age_star, mag_star, stack, n)
    
    # fix orbital parameters to 0
    e, o, t0 = 0., 0., 0.

    # sample i and w
    nsamples = 10000

    unif = np.random.uniform(size=nsamples)
    i = np.arccos(unif)*(180./np.pi)

    w = np.random.uniform(size=nsamples, low=0, high=360.)

    # for a and m_p
    del_m_p = 3.0
    del_a = 2.
    m_p_arr = np.arange(10, 50, del_m_p)
    a_arr = np.arange(np.min(seps)*d_star, np.max(seps)*d_star,del_a)

    # set up plot
    fig = plt.figure(figsize=(12,5))
    ax1=plt.subplot(111)
    norm = colors.Normalize(vmin=0,vmax=100)
    scalar_map = cm.ScalarMappable(norm=norm,cmap=cm.Blues)

    fig = plt.figure(figsize=(12,5))
    ax2=plt.subplot(111)
    
    fig = plt.figure(figsize=(12,5))
    ax3=plt.subplot(111)
    
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

            if test:
                if a == a_arr[10] and m_p == m_p_arr[0]:
                    plt.figure()
                    plt.hist(rv_diff)
                    plt.figure()
                    plt.errorbar(t_rv,rv_rv[:,idx_notdetected_rv[10]],marker='o',yerr=rv_std)

            # get frac detected either or
            idx_detected_rvorhci = np.where( (rv_diff > 5*rv_std) | (m_p > mass_lims) )[0]
            frac_detected_rvorhci = len(idx_detected_rvorhci)/float(nsamples)
                    
            # plot imaging
            rect = Rectangle( (a-del_a/2.,m_p-del_m_p/2.),
                              del_a,del_m_p,
                              facecolor=scalar_map.to_rgba(frac_detected_hci*100.),edgecolor='black')
            ax1.add_patch(rect)
            ax1.scatter(a,m_p)

            ax1.text(a, m_p, '{:2.0f}'.format(frac_detected_hci*100.),
                     verticalalignment='center',horizontalalignment='center')

            # plot RV
            rect = Rectangle( (a-del_a/2.,m_p-del_m_p/2.),
                              del_a,del_m_p,
                              facecolor=scalar_map.to_rgba(frac_detected_rv*100.),edgecolor='black')
            ax2.add_patch(rect)
            ax2.scatter(a,m_p)
            
            ax2.text(a, m_p, '{:2.0f}'.format(frac_detected_rv*100.),
                     verticalalignment='center',horizontalalignment='center')

            # plot both!
            rect = Rectangle( (a-del_a/2.,m_p-del_m_p/2.),
                              del_a,del_m_p,
                              facecolor=scalar_map.to_rgba(frac_detected_rvorhci*100.),edgecolor='black')
            ax3.add_patch(rect)
            ax3.scatter(a,m_p)
            
            ax3.text(a, m_p, '{:2.0f}'.format(frac_detected_rvorhci*100.),
                     verticalalignment='center',horizontalalignment='center')

            # ultimately: frac detected by one method OR the other
            
    plt.show()        
                            
    # get max delta v in the rv time window

    #plt.figure()
    #plt.xlabel('Projected separation')
    #plt.hist((x_ast.flatten()**2. + y_ast.flatten()**2.)**0.5)
    #plt.figure()
    #plt.xlabel('RV change')
    #plt.hist(rv_rv[0] - rv_rv[1])
    #plt.show()
