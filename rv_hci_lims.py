import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.io import ascii

import orbits.kepler as k

def sample_orbits():

    # get star info
    star = 'hd_36395'
    star_info = ascii.read('/Users/annaboehle/research/code/directimaging_code/plotting/nearest_solar_dist_age.txt')
    row = np.where(star_info['object_name'] == star)[0][0]

    d_star = star_info['dist'][row]
    age_star = (star_info['age_min'][row] + star_info['age_max'][row])/2.  # change this to actual best estimate for age
    mag_star = star_info['mag_star'][row]
    t_baseline = star_info['t_baseline'][row]
    m_star = star_info['m_star'][row]
    rv_std = star_info['rv_std'][row]
    
    t_rv = np.array([420.0041, 6640.9533]) + 2450000.
    t_hci = np.array([2454786.73611])

    # fix orbital parameters to 0
    e, o, t0 = 0., 0., 0.

    # sample i and w
    nsamples = 10000

    unif = np.random.uniform(size=nsamples)
    i = np.arccos(unif)*(180./np.pi)

    w = np.random.uniform(size=nsamples, low=0, high=360.)

    # for a and m_p
    a = 21. # AU
    m_p = 5. # M_J

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

    print x_rv.shape
    
    # find astrometric location for these parameters
    x_ast, y_ast, rv_ast = k.orbit_solution(E_hci,
                                            e=e, P=P, t0=t0, i=i, w=w, o=o, m_p=m_p, m_star=m_star,d_star=d_star,
                                            units = 'arcsec',
                                            solve_kepler=False)
    print x_ast.shape


    # get projected separations
    sep = (x_ast.flatten()**2. + y_ast.flatten()**2.)**0.5

    # get max delta v in the rv time window

    plt.figure()
    plt.xlabel('Projected separation')
    plt.hist((x_ast.flatten()**2. + y_ast.flatten()**2.)**0.5)
    plt.figure()
    plt.xlabel('RV change')
    plt.hist(rv_rv[0] - rv_rv[1])
    plt.show()
