import numpy as np
import matplotlib.pyplot as plt

class Star:
    """
    :param name: name of star (for printing)
    :type name: str
    :param dist: distance to star in pc
    :type dist: float
    :param age: age of star in Gyr
    :type age: float
    :param mags: magnitudes of star in range of bands
    :type mags: dict
    :param mass: mass of star in solar masses
    :type mass: float
    """

    def __init__(self,
                 name,
                 dist,
                 age,
                 mags,
                 mass):

        self.name = name
        self.dist = dist
        self.age = age
        self.mags = mags
        self.mass = mass

    def set_mag(self,
                band,
                mag,
                overwrite=False):

        if band not in self.mags.keys():
            self.mags[band] = mag

        else:

            if overwrite:
                self.mags[band] = mag

            else:
                raise ValueError(f'{band} is already set to {self.mags[band]} mag in dictionary of stellar magnitudes.\n' + \
                                 f'Set overwrite = True to overwrite this value.')


class Planet:
    """
    :param mass: mass of planet in Jupiter masses
    """

    def __init__(self,
                 mass):

        self.mass = mass


    # need some function to get flux!

class Orbit:
    """
    :param e: orbital eccentricity
    :type e: float or np.ndarray of floats
    :param p: orbital period in days
    :type p: float or np.ndarray of floats
    :param t0: time of closest approach in days
    :type t0: float or np.ndarray of floats
    :param i: orbital inclination in degrees
    :type i: float or np.ndarray of floats
    :param w: argument of periapse in degrees (little omega)
    :type w: float or np.ndarray of floats
    :param o: angle of ascending node in degrees (big omega)
    :type o: float or np.ndarray of floats
    :param star: Star instance
    :type star: Star
    :param planet: Planet instance
    :type planet: Planet
    """

    def __init__(self,
                 e,
                 p,
                 t0,
                 i,
                 w,
                 o,
                 star,
                 planet):

        self.e = e
        self.p = p
        self.t0 = t0

        # convert angles to radians
        self.i = i * (np.pi / 180.)
        self.w = w * (np.pi / 180.)
        self.o = o * (np.pi / 180.)

        self.star = star
        self.planet = planet

        # should I check that any inputted arrays are the same length?
        #if any([type(param) is )

    def true_anomaly(self,
                     ecc_anomaly):
        """
        :param ecc_anomaly: Array of eccentric anomalies in radians
        :type ecc_anomaly: np.ndarray

        :return: true anomaly
        :rtype: np.ndarray
        """

        f = 2 * np.arctan(np.sqrt((1 + self.e) / (1 - self.e)) * np.tan(ecc_anomaly / 2.))

        return f

    def solve_kepler_eqn(self,
                         times,
                         tol=1e-12,
                         test=False):
        """
        Solve Kepler's equation using Newton's method from: http://mathworld.wolfram.com/KeplersEquation.html

        :param times: Array of times in JD (same length as E)
        :type times: np.ndarray
        :param tol: tolerance of solving Kepler's equation (solved with kepler_eqn returns =< tol)
        :type tol: float
        :param test: if True, plots E values versus the kepler equation to check solutions.
        :type test: bool

        :return: eccentric anomalies for each input time in the array t
        :rtype: ndarray
        """

        # array of initial guesses for E's: use the mean anomaly
        # E = np.zeros(len(t))
        E = (2 * np.pi / self.p) * (times - self.t0)

        def kepler_eqn(ecc_anomaly, t):

            return ecc_anomaly - self.e * np.sin(ecc_anomaly) - (2 * np.pi / self.p) * (t - self.t0)

        # difference between f(E_init)
        tol_check = kepler_eqn(E, times)

        # iterate until all times have converged
        niter = 0
        while (np.abs(tol_check) > tol).any():
            # first order
            E = E - (kepler_eqn(E, times) / (1 - self.e * np.cos(E)))

            # second order
            # E = E - ( (1 - e*np.cos(E)) / (e*np.sin(E)) )

            tol_check = kepler_eqn(E, times)

            niter += 1

        if test:
            E_arr = np.arange(-np.pi, np.pi, 0.1)
            for i, t_val in enumerate(t):
                line = plt.plot(E_arr, kepler_eqn(E_arr, t_val))
                plt.axvline(E[i], color=line[0].get_color())
            plt.axhline(0.0, color='black', linestyle='dotted')

            print(niter)

        return E

    def get_orbit_solution(self,
                           times,
                           units='arcsec'):

        # convert masses to solar masses
        m_p_solar = (self.planet.mass / (const.M_sun / const.M_jup)).value

        # convert p to a (semi-major axis for relative orbit - for astrometric orbit)
        a = ((self.star.mass + m_p_solar) * (self.p / 365.) ** 2.) ** (1. / 3)  # in AU

        if units == 'arcsec':
            a = a / self.star.dist  # in arcsec
        elif units != 'AU':
            ValueError('ERROR: units parameters must be "AU" or "arcsec"')

        # get eccentric anomaly and true anomaly
        #if solve_kepler:
        E = self.solve_kepler_eqn(times)
        #else:
        #    E = t_or_E

        f = self.true_anomaly(E, self.e)

        # Thiele-Innes constants
        A = a * (np.cos(self.w) * np.cos(self.o) - np.sin(self.w) * np.cos(self.i) * np.sin(self.o))
        F = a * (-np.sin(self.w) * np.cos(self.o) - np.cos(self.w) * np.cos(self.i) * np.sin(self.o))
        B = a * (np.cos(self.w) * np.sin(self.o) + np.sin(self.w) * np.cos(self.i) * np.cos(self.o))
        G = a * (-np.sin(self.w) * np.sin(self.o) + np.cos(self.w) * np.cos(self.i) * np.cos(self.o))

        K = (28.4329 / (np.sqrt(1 - self.e ** 2.))) * (self.planet.mass * np.sin(self.i)) \
            * (m_p_solar + self.star.mass) ** (-2. / 3) \
            * (self.p / 365.) ** (-1. / 3)

        if isinstance(A, np.ndarray) and isinstance(E, np.ndarray):
            len_A = len(A)
            len_E = len(E)

            # tile matrix elements
            A = np.tile(A, [len_E, 1])
            F = np.tile(F, [len_E, 1])
            B = np.tile(B, [len_E, 1])
            G = np.tile(G, [len_E, 1])

            # tile time parameters for next calculations
            E = np.tile(np.array([E]).transpose(), [1, len_A])
            f = np.tile(np.array([f]).transpose(), [1, len_A])

        X = np.cos(E) - self.e
        Y = (1 - self.e ** 2.) ** 0.5 * np.sin(E)

        # flipped x and y so the reference direction is along +x direction as in Fig.7 of exoplanets ch. 2
        return (A * X + F * Y), (B * X + G * Y), K * (np.cos(self.w + f) + self.e * np.cos(self.w))
