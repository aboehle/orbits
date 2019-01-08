import orbits.kepler as k
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt

def check_orbits(star = 'hd_36395',sep = 3.51613600, test=False):

    star_info = ascii.read('/Users/annaboehle/research/code/directimaging_code/plotting/nearest_solar_dist_age.txt')
    row = np.where(star_info['object_name'] == star)[0][0]

    d_star = star_info['dist'][row]
    age_star = [star_info['age_min'][row],star_info['age_max'][row]]
    mag_star = star_info['mag_star'][row]    
    t_baseline = star_info['t_baseline'][row]
    m_star = star_info['m_star'][row]
    rv_std = star_info['rv_std'][row]

    t_rv_abs = np.array([420.0041, 6640.9533]) + 2450000.
    t_ast_abs = 2454786.73611
    t_rv_rel = t_rv_abs - t_ast_abs
    
    sep_AU = sep*d_star
    a_range = np.linspace(21.,80.,10) #np.arange(sep_AU, 50., 15.)

    m_p_range = [17.6415733225, 31.2563360693] # M_jup

    n_incl = 10
 
    delta_rv_min = np.zeros((len(a_range), 2, n_incl))
    delta_rv_max = np.zeros((len(a_range), 2, n_incl))    

    e = 0.0
    
    for a_idx, a in enumerate(a_range):
        
        for m_idx, m_p in enumerate(m_p_range):

            # convert to planet a and period
            p = np.sqrt(a**3./(m_star+m_p*954.7919e-6))*365.

            # find minimum inclination that could have the detected separation: i must be between 90 deg and this
            i_min = (np.pi/2 - np.arcsin((sep*d_star)/a))*(180./np.pi)  # hmm, I'm not sure which a should go in here...

            i_range = np.linspace(i_min+0.5, 90., n_incl) #np.arange(i_min, 90., 5.)  # degrees

            if test:
                print i_min
                print i_range

            for i_idx, i in enumerate(i_range):
                roots = k.t_from_angularsep(sep, e=e, P=p, t0=0., i=i, w=0.,
                                            m_p=m_p, m_star=m_star, d_star=d_star, test=test)
                
                t_rv = np.tile(np.array([t_rv_rel]),(len(roots),1)) + np.tile(roots,(2,1)).transpose() + 2450000.

                # for the time ranges and the roots, get the orbital measurements
                t_check = np.concatenate( (np.ravel(t_rv), roots + 2450000.))

                x_check, y_check, rv_check = k.orbit_solution(t_check - 2450000.,
                                                              e=e, P=p, t0=0, i=i, w=0., o=0., m_p=m_p, m_star=m_star,
                                                              d_star=d_star,
                                                              units = 'arcsec')
                

                # only need rv diff, because the RV baseline is much shorter than the orbital period for a > sep_AU ~ 20
                # i.e., no chance of getting a turning point!

                if test:
                    t = np.arange(0, p, 0.1)
                    x, y, rv = k.orbit_solution(t, e=e, P=p, t0=0, i=i, w=0., o=0., m_p=m_p, m_star=m_star, d_star=d_star, units = 'arcsec')
                    plt.figure()
                    plt.plot(t + 2450000., rv)
                    for time in t_rv:
                        plt.axvline(time[0])
                        plt.axvline(time[1])
                    plt.scatter(t_check[0:t_rv.size], rv_check[0:t_rv.size])
                    plt.show()
                    
                    print rv_check

                delta_rv_tmp = np.zeros(len(roots))
                c = 0
                
                for idx in range(0, t_rv.size, 2):

                    delta_rv_tmp[c] = np.abs(rv_check[0:t_rv.size][idx] - rv_check[0:t_rv.size][idx+1])
                    c += 1                    
                    
                delta_rv_min[a_idx, m_idx, i_idx] = np.min(delta_rv_tmp)
                delta_rv_max[a_idx, m_idx, i_idx] = np.max(delta_rv_tmp)                

    return delta_rv_min, delta_rv_max
