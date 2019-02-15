import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import scipy


def kepler_eqn(E, t, e, P, t0):
    """
    Kepler equation.

    :param E: Array of eccentric anomalies in radians
    :type E: ndarray
    :param t: Array of times in JD (same length as E)
    :type t: ndarray
    :param e: orbital eccentricity
    :type e: float
    :param P: orbital period in days
    :type P: float
    :param t0: time of closest approach in JD
    :type t0: float


    :return: result of Kepler's equation (should be 0 when the eqn is solved).
    :rtype: float
    """

    #try:
    return E - e*np.sin(E) - (2*np.pi/P)*(t - t0)
    #except:
        

def solve_kepler_eqn(t, e, P, t0, tol=1e-12, test=False):
    """
    Solve Kepler's equation using Newton's method from: http://mathworld.wolfram.com/KeplersEquation.html

    :param t: Array of times in JD (same length as E)
    :type t: ndarray
    :param e: orbital eccentricity
    :type e: float
    :param P: orbital period in days
    :type P: float
    :param t0: time of closest approach in JD
    :type t0: float
    :param tol: tolerance of solving Kepler's equation (solved with kepler_eqn returns =< tol)
    :type tol: float
    :param test: if True, plots E values versus the kepler equation to check solutions.
    :type test: Boolean

    :return: eccentric anomalies for each input time in the array t
    :rtype: ndarray
    """

    # array of initial guesses for E's: use the mean anomaly
    #E = np.zeros(len(t))
    E = (2*np.pi/P)*(t - t0)

    # difference between f(E_init)
    tol_check = kepler_eqn(E, t, e, P, t0)

    # iterate until all times have converged
    niter = 0
    while (np.abs(tol_check) > tol).any():

        # first order
        E = E - (kepler_eqn(E, t, e, P, t0)/(1 - e*np.cos(E)))

        # second order
        # E = E - ( (1 - e*np.cos(E)) / (e*np.sin(E)) )

        tol_check = kepler_eqn(E, t, e, P, t0)

        niter += 1

    if test:
        E_arr = np.arange(-np.pi, np.pi, 0.1)
        for i, t_val in enumerate(t):
            line = plt.plot(E_arr, kepler_eqn(E_arr, t_val, e, P, t0))
            plt.axvline(E[i], color=line[0].get_color())
        plt.axhline(0.0,color='black',linestyle='dotted')

        print niter
    
    return E


def true_anomaly(E, e):
    """
    :param E: Array of eccentric anomalies in radians
    :type E: ndarray
    :param e: orbital eccentricity
    :type e: float

    :return: true anomaly
    :rtype: ndarray
    """

    f = 2*np.arctan(np.sqrt((1 + e) / (1 - e))*np.tan(E/2.))

    return f


def orbit_3d(t, e, P, t0, i, w, o, m_p, m_star):
    """

    :param t: Array of times in JD (same length as E)
    :type t: ndarray
    :param e: orbital eccentricity
    :type e: float
    :param P: orbital period in days
    :type P: float
    :param t0: time of closest approach in JD
    :type t0: float
    :param i: orbital inclination in degrees
    :type i: float
    :param w: argument of periapse in degrees (little omega)
    :type w: float
    :param o: angle of ascending node (big omega)
    :type o: float
    :param m_p: mass of planet in Jupiter masses
    :type m_p: float
    :param m_star: mass of star in solar masses
    :type m_star: float

    :return: x, y, z (at each of the input times t, all in AU)
    :rtype: ndarray, ndarray, ndarray

    """

    # convert masses to solar masses
    m_p = m_p/(const.M_sun/const.M_jup)
    
    # convert p to a
    a = ((m_star + m_p)*(P/365.)**2.)**(1./3)  # in AU
    
    # convert angles to radians
    i, w, o = i*(np.pi/180.), w*(np.pi/180.), o*(np.pi/180.)

    # get eccentric anomaly
    E = solve_kepler_eqn(t, e, P, t0)
    f = true_anomaly(E, e)

    r = a*(1 - e*np.cos(E))  # in AU

    # project onto coordinate axes
    x = r*(np.cos(o)*np.cos(f + w) - np.sin(o)*np.sin(f + w)*np.cos(i))
    y = r*(np.sin(o)*np.cos(f + w) + np.cos(o)*np.sin(f + w)*np.cos(i))
    z = r*(np.sin(f + w)*np.sin(i))

    return x, y, z


def orbit_solution(t_or_E, e, P, t0, i, w, o, m_p, m_star, d_star, units = 'arcsec', solve_kepler = True):
    """
    :param t_or_E:
    :type t_or_E: ndarray
    :param e: orbital eccentricity
    :type e: float
    :param P: orbital period in days
    :type P: float
    :param t0: time of closest approach in JD
    :type t0: float
    :param i: orbital inclination in degrees
    :type i: float
    :param w: argument of periapse in degrees (little omega)
    :type w: float
    :param o: angle of ascending node (big omega)
    :type o: float
    :param m_p: mass of planet in Jupiter masses
    :type m_p: float
    :param m_star: mass of star in solar masses
    :type m_star: float
    :param d_star: distance to star in pc
    :type d_star: float
    :param units: units of the output x/y orbital positions ('arcsec' or 'AU')
    :type units: string
    :param solve_kepler: if True, then input of t_or_E is taken as the times (t) and kepler's eqn is solved to get E.
                         if False, then input of t_or_E is taken as the eccentric anomalies (E).
    :type solve_kepler: bool

    :return: x, y, RV (predicted orbital positions on the plane of the sky - x,y in units above - and predicted stellar radial velocities)
    :rtype ndarray, ndarray, ndarray
    """

    # convert masses to solar masses
    m_p_solar = (m_p/(const.M_sun/const.M_jup)).value
    
    # convert p to a (semi-major axis for relative orbit - for astrometric orbit)
    a = ((m_star + m_p_solar)*(P/365.)**2.)**(1./3)  # in AU
    
    if units == 'arcsec':
        a = a/d_star  # in arcsec
    elif units != 'AU':
        ValueError('ERROR: units parameters must be "AU" or "arcsec"')

    # convert angles to radians
    i, w, o = i*(np.pi/180.), w*(np.pi/180.), o*(np.pi/180.)

    # get eccentric anomaly and true anomaly
    if solve_kepler:
        E = solve_kepler_eqn(t_or_E, e, P, t0)
    else:
        E = t_or_E
        
    f = true_anomaly(E, e)

    # Thiele-Innes constants
    # A = a*(np.cos(o)*np.cos(w) - np.sin(o)*np.sin(w)*np.cos(i))
    # B = a*(np.sin(o)*np.cos(w) + np.cos(o)*np.sin(w)*np.cos(i))
    # F = a*(-np.cos(o)*np.sin(w) - np.sin(o)*np.cos(w)*np.cos(i))
    # G = a*(-np.sin(o)*np.sin(w) + np.cos(o)*np.cos(w)*np.cos(i))
    # print A, B, F, G

    # rotation matrices (Wright and Howard 2009)
    # -> doing multiplication separately to allow for array inputs of orbital parameters
    # R_z_o = np.matrix([[np.cos(o), np.sin(o), 0],
    #                 [-np.sin(o), np.cos(o), 0],
    #                 [0, 0, 1.]])
    # R_z_w = np.matrix([[np.cos(w), np.sin(w), 0],
    #                 [-np.sin(w), np.cos(w), 0],
    #                 [0, 0, 1]])
    # R_x_i = np.matrix([[1., 0, 0],
    #                 [0, np.cos(i), -np.sin(i)],
    #                 [0, np.sin(i), np.cos(i)]])
    # mat_out = a*(R_z_w*(R_x_i*R_z_o))

    # A, B, F, G = mat_out[0,0], mat_out[0,1], mat_out[1,0], mat_out[1,1]

    A = a*(np.cos(w)*np.cos(o) - np.sin(w)*np.cos(i)*np.sin(o))
    F = a*(-np.sin(w)*np.cos(o) - np.cos(w)*np.cos(i)*np.sin(o))
    B = a*(np.cos(w)*np.sin(o) + np.sin(w)*np.cos(i)*np.cos(o))
    G = a*(-np.sin(w)*np.sin(o) + np.cos(w)*np.cos(i)*np.cos(o))

    K = (28.4329/(np.sqrt(1-e**2.)))*(m_p*np.sin(i))*(m_p_solar + m_star)**(-2./3)*(P/365.)**(-1./3)

    if isinstance(A, np.ndarray) and isinstance(E, np.ndarray):
        len_A = len(A)
        len_E = len(E)

        # tile matrix elements
        A = np.tile(A, [len_E,1])
        F = np.tile(F, [len_E,1])
        B = np.tile(B, [len_E,1])
        G = np.tile(G, [len_E,1])

        # tile time parameters for next calculations
        E = np.tile(np.array([E]).transpose(), [1, len_A])
        f = np.tile(np.array([f]).transpose(), [1, len_A])
        
    X = np.cos(E) - e
    Y = (1 - e**2.)**0.5*np.sin(E)

    # flipped x and y so the reference direction is along +x direction as in Fig.7 of exoplanets ch. 2 
    return (A*X + F*Y), (B*X + G*Y), K*(np.cos(w+f) + e*np.cos(w))


def angular_sep_f(f, e, P, i, w, m_p, m_star, d_star):
    """
    Find angular separation versus the true anomaly for a give set of orbital parameters.

    :param f: array of true anomalies
    :type f: ndarray
    :param e: orbital eccentricity
    :type e: float
    :param P: orbital period in days
    :type P: float
    :param i: orbital inclination in degrees
    :type i: float
    :param w: argument of periapse in degrees (little omega)
    :type w: float
    :param m_p: mass of planet in Jupiter masses
    :type m_p: float
    :param m_star: mass of star in solar masses
    :type m_star: float
    :param d_star: distance to star in pc
    :type d_star: float

    :return: angular separation in arcsec for all input true anomaly values
    :rtype: ndarray
    """
    
    # convert period to relative orbit semi-major axis
    m_p_solar = (m_p/(const.M_sun/const.M_jup)).value    
    a = ((m_star + m_p_solar)*(P/365.)**2.)**(1./3)  # in AU

    # get separation
    r = (a*(1 - e**2.))/(1 + e*np.cos(f))

    # convert angles to radians
    i, w = i*(np.pi/180.), w*(np.pi/180.)

    return (r/d_star)*np.sqrt( (np.cos(w + f))**2. + (np.sin(w + f))**2.*(np.cos(i))**2.)


def angular_sep(t, e, P, t0, i, w, m_p, m_star, d_star):
    """
    Find angular separation versus time for a give set of orbital parameters.

    :param t: array of times
    :type t: ndarray
    :param e: orbital eccentricity
    :type e: float
    :param P: orbital period in days
    :type P: float
    :param t0: time of closest approach in JD
    :type t0: float
    :param i: orbital inclination in degrees
    :type i: float
    :param w: argument of periapse in degrees (little omega)
    :type w: float
    :param m_p: mass of planet in Jupiter masses
    :type m_p: float
    :param m_star: mass of star in solar masses
    :type m_star: float
    :param d_star: distance to star in pc
    :type d_star: float

    :return: angular separation in arcsec for all input times
    :rtype: ndarray
    """
    
    E = solve_kepler_eqn(t, e, P, t0)
    f = true_anomaly(E, e)
    
    # convert period to relative orbit semi-major axis
    m_p_solar = (m_p/(const.M_sun/const.M_jup)).value    
    a = ((m_star + m_p_solar)*(P/365.)**2.)**(1./3)  # in AU

    # get separation
    r = (a*(1 - e**2.))/(1 + e*np.cos(f))  # in AU

    # convert angles to radians
    i, w = i*(np.pi/180.), w*(np.pi/180.)

    return (r/d_star)*np.sqrt((np.cos(w + f))**2. + ((np.sin(w + f))*np.cos(i))**2.)


def f_from_angularsep(sep_meas, e, P, i, w, m_p, m_star, d_star, test=False):
    """
    Find the true anomalies (can be multiple) for a given angular separation and set of orbital parameters.

    :param sep_meas: measured separation of planet in arcsec
    :type sep_meas: float
    :param e: orbital eccentricity
    :type e: float
    :param P: orbital period in days
    :type P: float
    :param i: orbital inclination in degrees
    :type i: float
    :param w: argument of periapse in degrees (little omega)
    :type w: float
    :param m_p: mass of planet in Jupiter masses
    :type m_p: float
    :param m_star: mass of star in solar masses
    :type m_star: float
    :param d_star: distance to star in pc
    :type d_star: float
    :param test: if True, plots the angular separation versus true anomaly for the input orbit and shows
                 the true anomalies corresponding to the input measured separation.
    :type test: boolean

    :return: true anomalies at which the planet has the inputted angular separation
    :rtype: ndarray

    """

    func_theta = lambda f, sep_meas, e, P, i, w, m_p, m_star, d_star: angular_sep_f(f, e, P, i, w, m_p, m_star, d_star) - sep_meas

    # check how many roots there are
    f_range = np.arange(0, 2*np.pi, 0.01)
    theta = func_theta(f_range, sep_meas, e, P, i, w, m_p, m_star, d_star)
    theta_sign = np.sign(theta)
    diff = theta_sign[1:] - theta_sign[0:-1]
    
    if len(np.where(diff != 0)[0]) >= 1:
        root = scipy.optimize.fsolve(func_theta, f_range[np.where(diff != 0)], args=(sep_meas, e, P, i, w, m_p, m_star, d_star))
    else:
        raise ValueError('For inputted orbital parameters, angular separation is never reached!')

    # shift root to 0 -> 2*pi
    div = np.floor_divide(root,(2*np.pi))
    root = root - div*2*np.pi

    if test:
        print 'true anomaly:', root

        plt.plot(f_range, func_theta(f_range, sep_meas, e, P, i, w, m_p, m_star, d_star))
        for r in root:
            plt.axvline(r)
        plt.axhline(0.0)
        plt.xlabel('True anomaly (radians)')
        plt.ylabel('Angular sep - measured sep')        
        plt.show()
    
    return root


def t_from_angularsep(sep_meas, e, P, t0, i, w, m_p, m_star, d_star, test=True):
    """
    Find the times (can be multiple) for a given angular separation and set of orbital parameters.

    :param sep_meas: measured separation of planet in arcsec
    :type sep_meas: float
    :param e: orbital eccentricity
    :type e: float
    :param P: orbital period in days
    :type P: float
    :param t0: time of closest approach in JD
    :type t0: float
    :param i: orbital inclination in degrees
    :type i: float
    :param w: argument of periapse in degrees (little omega)
    :type w: float
    :param m_p: mass of planet in Jupiter masses
    :type m_p: float
    :param m_star: mass of star in solar masses
    :type m_star: float
    :param d_star: distance to star in pc
    :type d_star: float
    :param test: if True, plots the angular separation versus true anomaly for the input orbit and shows
                 the true anomalies corresponding to the input measured separation.
    :type test: boolean

    :return: times at which the planet has the inputted angular separation
    :rtype: ndarray
    """

    func_theta = lambda t, sep_meas, e, P, t0, i, w, m_p, m_star, d_star: angular_sep(t, e, P, t0, i, w, m_p, m_star, d_star) - sep_meas

    # check how many roots there are
    t_range = np.arange(0, P, 0.1)
    theta = func_theta(t_range, sep_meas, e, P, t0, i, w, m_p, m_star, d_star)
    theta_sign = np.sign(theta)
    diff = theta_sign[1:] - theta_sign[0:-1]

    if test:
        plt.plot(t_range, theta)
        
    if len(np.where(diff != 0)[0]) >= 1:
        root = scipy.optimize.fsolve(func_theta, t_range[np.where(diff != 0)], args=(sep_meas, e, P, t0, i, w, m_p, m_star, d_star))
    else:
        raise ValueError('For inputted orbital parameters, angular separation is never reached!')

    # shift root to 0 -> P
    div = np.floor_divide(root,(P))
    root = root - div*P
    
    if test:
        print 't:',root
        
        plt.plot(t_range, theta)
        for r in root:
            plt.axvline(r)
        plt.axhline(0.0)
        plt.xlabel('Time (days)')
        plt.ylabel('Angular sep - measured sep')
        plt.show()
    
    return root
