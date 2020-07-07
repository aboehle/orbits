import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import orbits.hci_util as util
from scipy.interpolate import interp1d, griddata

class Star:
    """
    :param name: name of star (for printing)
    :type name: str
    :param dist: distance to star in pc
    :type dist: float
    :param age: tuple of 3 floats giving (minimum, nominal, and maximum) ages of star in Gyr
    :type age: tuple
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
                 mass,
                 teff,
                 radius):

        self.name = name
        self.dist = dist
        self.age = age
        self.mags = mags
        self.mass = mass
        self.teff = teff
        self.radius = radius

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

    def get_age(self,
                age_type):
        """

        :param age_type: 'min', 'nominal', or 'max' age
        :type age_type: str

        :return: specified age of star
        :rtype: float
        """

        if age_type == 'min':
            return self.age[0]
        elif age_type == 'nominal':
            return self.age[1]
        elif age_type == 'max':
            return self.age[2]
        else:
            raise ValueError("'age_type' must be 'min', 'nominal', or 'max'.")


class Planet:
    """
    :param mass: mass of planet in Jupiter masses
    :param radius: radius of planet in Earth radii
    """

    def __init__(self,
                 mass=None,
                 radius=None):

        if mass is None and radius is None:
            raise ValueError("Either planet mass or radius must be defined.")

        elif radius is None:
            self.mass = mass
            self.radius = util.mass_radius_relationship([mass*const.M_jup/const.M_earth])[0]

        elif mass is None:
            self.radius = radius
            self.mass = util.radius_mass_relationship([radius])[0]*const.M_earth/const.M_jup

        else:
            self.mass = mass
            self.radius = radius


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

        self.e = np.asarray(e)
        self.p = np.asarray(p)
        self.t0 = np.asarray(t0)

        # convert angles to radians
        self.i = np.asarray(np.asarray(i) * (np.pi / 180.))
        self.w = np.asarray(np.asarray(w) * (np.pi / 180.))
        self.o = np.asarray(np.asarray(o) * (np.pi / 180.))

        self.star = star
        self.planet = planet

        # check whether inputted arrays, if any, are same length
        # (i.e., check that all params with length != 1 have the same size)
        s = np.asarray([param.size for param in [self.e, self.p, self.t0, self.i, self.w, self.o]])

        if 0 in s:
            raise ValueError("Orbital parameters must be floats or iterators that have a length > 0.")

        arr_sizes = s[np.where(s != 1)]

        if len(np.unique(arr_sizes)) > 1:
            raise ValueError("Orbital parameters that were given as an iterator do not have the same lengths.")

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
                           times=None,
                           units='arcsec',
                           exact_total_mass=True):
        """

        :param times:
        :param units: units of output (x, y, AND z)
        :return:
        """

        # convert masses to solar masses
        if exact_total_mass:
            m_p_solar = (self.planet.mass / (const.M_sun / const.M_jup)).value
        else:
            m_p_solar = 0.0

        # convert p to a (semi-major axis for relative orbit - for astrometric orbit)
        a = ((self.star.mass + m_p_solar) * (self.p / 365.) ** 2.) ** (1. / 3)  # in AU

        if units == 'arcsec':
            a = a / self.star.dist  # in arcsec
        elif units != 'AU':
            ValueError('ERROR: units parameters must be "AU" or "arcsec"')

        # get eccentric anomaly and true anomaly
        ecc_anomaly = self.solve_kepler_eqn(times)
        f = self.true_anomaly(ecc_anomaly)

        # Thiele-Innes constants
        A = a * (np.cos(self.w) * np.cos(self.o) - np.sin(self.w) * np.cos(self.i) * np.sin(self.o))
        F = a * (-np.sin(self.w) * np.cos(self.o) - np.cos(self.w) * np.cos(self.i) * np.sin(self.o))
        B = a * (np.cos(self.w) * np.sin(self.o) + np.sin(self.w) * np.cos(self.i) * np.cos(self.o))
        G = a * (-np.sin(self.w) * np.sin(self.o) + np.cos(self.w) * np.cos(self.i) * np.cos(self.o))

        K = (28.4329 / (np.sqrt(1 - self.e ** 2.))) * (self.planet.mass * np.sin(self.i)) \
            * (m_p_solar + self.star.mass) ** (-2. / 3) \
            * (self.p / 365.) ** (-1. / 3)

        if isinstance(A, np.ndarray) and isinstance(ecc_anomaly, np.ndarray):
            len_A = len(A)
            len_ecc_anomaly = len(ecc_anomaly)

            # tile matrix elements
            A = np.tile(A, [len_ecc_anomaly, 1])
            F = np.tile(F, [len_ecc_anomaly, 1])
            B = np.tile(B, [len_ecc_anomaly, 1])
            G = np.tile(G, [len_ecc_anomaly, 1])

            # tile time parameters for next calculations
            ecc_anomaly = np.tile(np.array([ecc_anomaly]).transpose(), [1, len_A])
            f = np.tile(np.array([f]).transpose(), [1, len_A])

        X = np.cos(ecc_anomaly) - self.e
        Y = (1 - self.e ** 2.) ** 0.5 * np.sin(ecc_anomaly)

        # get physical separation and z
        r = (a*(1 - self.e**2.))/(1 + self.e*np.cos(f))
        z = r*(np.sin(f + self.w)*np.sin(self.i))


        # flipped x and y so the reference direction is along +x direction as in Fig.7 of exoplanets ch. 2
        return (A * X + F * Y), (B * X + G * Y), z, K * (np.cos(self.w + f) + self.e * np.cos(self.w))


class RVDataSet:

    def __init__(self,
                 times,
                 rv,
                 star):

        self.times = times
        self.rv = rv
        self.star = star

        self.rv_std = np.std(self.rv, ddof=1)


class ContrastModel:

    def __init__(self,
                 model_type,
                 phoenix_model=None,
                 phoenix_inst=None,
                 ):

        if model_type not in ['teq', 'phoenix']:
            raise ValueError(
                f"'model_type' given was {model_type}: must be 'teq' for equilibrium temperature or 'phoenix'" + \
                                          f" for internal heat from the Phoenix models.")

        self.model_type = model_type

        # hmm doesn't make sense to have these defined for non-phoenix models...
        self.phoenix_model = 'AMES-Cond-2000'
        self.phoenix_inst = 'NaCo'

    def get_contrast(self,
                     planet,
                     star,
                     age_type=None,
                     filter=None,
                     r=None,
                     ):
        """

        :param planet:
        :param star:
        :param age_type:
        :param filter:
        :param r: physical separation between star and planet in AU
        :return:
        """

        if self.model_type == 'teq' and r is None:
            raise ValueError("r (physical separation between star and planet) must be given is model_type = 'teq'.")

        if self.model_type == 'phoenix':
            contrast = util.calc_contrast_phoenix(planet.mass,
                                                  star,
                                                  filter,
                                                  star.get_age(age_type),
                                                  model=self.phoenix_model,
                                                  inst=self.phoenix_inst)

        elif self.model_type == 'teq':
            contrast = util.calc_contrast_teq(planet.radius,
                                              r,
                                              star)
        return contrast


class ImagingDataSet:
    """
    :param time: time of imaging data set in ...
    :type time: float
    :param contrast: limiting contrasts for this data set in mag
    :type contrast: np.ndarray
    :param star: star that was observed
    :type star: Star
    :param filter: filter of observations
    :type filter: str
    :param sep: projected separations in arcsec of the contrast
    :type sep: np.ndarray
    :param pos: tuple of x and y positions of each pixel in 2D contrast, relative to star
    :type pos: tuple of np.ndarray
    """

    def __init__(self,
                 time,
                 contrast,
                 star,
                 filter,
                 sep=None,
                 pos=None):


        self.times = np.array([time])  # convert to right units!
        self.star = star

        self.sep = sep
        self.pos = pos
        self.contrast = contrast

        self.filter = filter

        self.ndim = contrast.ndim

        self.f_contrast = self._interpolate_contrast()

        if self.ndim == 1:
            if sep is None:
                raise ValueError("For 1D 'contrast', 'sep' must be defined.")
            elif sep.shape != contrast.shape:
                raise ValueError(f"'contrast' and 'sep' must have different shapes: {contrast.shape} and {sep.shape}.")

        elif self.ndim == 2:
            if pos is None:
                raise ValueError("For 2D 'contrast', 'pos' must be defined.")
                # do things...
                #f_masslimvsep = scipy.interpolate.interp1d(sep, contrast, bounds_error=False, fill_value=np.inf)
        else:
            raise ValueError(f"'contrast' must have either 1 or 2 dimensions.")

    @staticmethod
    def get_sep_rho(contrast_map,
                    pixscale):
        pass
        # return sep, rho 2d with same size as input contrast map
        # this can then be passed into the constructor

    def _interpolate_contrast(self):
        # could also go in constructor?
        # could change so that function always takes sep and rho, but just returns the same value for all rhos

        if self.contrast.ndim == 1:

            f_contrast = interp1d(self.sep,
                                  self.contrast,
                                  bounds_error=False,
                                  fill_value=-np.inf)

        else:
            idx = np.where(self.contrast != self.contrast)
            self.contrast[idx] = -np.inf

            f_contrast = lambda x, y: griddata((self.pos[0].flatten(),self.pos[1].flatten()),
                                               self.contrast.flatten(),
                                               (x,y),
                                               method='nearest')

        return f_contrast

        # else:
        #     if contrast.ndims != 1:
        #         raise ValueError("Contrast must be 1d if only seps are given and not rho.")



class CompletenessMC:

    def __init__(self,
                 mp_arr,
                 a_arr,
                 nsamples,
                 data_sets,
                 model=None):
        """

        :param mp_arr: array of planet masses in Jupiter masses
        :param a_arr: array of semi-major axes in AU
        :param nsamples: number of MC samples per planet mass/semi-major axis pair
        """

        self.mp_arr = np.asarray(mp_arr)
        self.a_arr = np.asarray(a_arr)

        self.data_sets = data_sets
        self.star = data_sets[0].star

        self.nsamples = nsamples

        self.model = model

        # check that the data sets all have the same star
        if len(data_sets) == 0:
            raise ValueError("Must include at least one data set (ImagingDataSet or RVDataSet).")

        elif len(data_sets) > 1:
            for data in data_sets[1:]:
                if data.star != self.star:
                    raise ValueError(f"data set {data} does not have the same star as first data set {data_sets[0]}")

        # sample orbital parameters

        # set fixed orbital parameters to 0
        self.e, self.o, self.t0 = 0., 0., 0.  # self.o needs to be sampled for 2d contrast map

        # sample i and w
        unif = np.random.uniform(size=self.nsamples)
        self.i = np.arccos(unif) * (180. / np.pi)

        self.w = np.random.uniform(size=self.nsamples, low=0, high=360.)

        # another check: that any imaging sets filters have corresponding magnitudes in the Star instances


    def run(self,
            exact_total_mass=True):
        """
        Derive the completeness map for each planet mass/semi-major axis combination.

        :return: completeness_map (array for each combination of values in mp_arr and a_arr,
        """

        if len(self.data_sets) > 1:
            completeness_map = np.zeros((len(self.data_sets) + 2,len(self.mp_arr),len(self.a_arr)))
        else:
            completeness_map = np.zeros((len(self.data_sets),len(self.mp_arr), len(self.a_arr)))

        for i, m_p in enumerate(self.mp_arr):
            planet = Planet(m_p)

            for j, a in enumerate(self.a_arr):

                if exact_total_mass:
                    p = np.sqrt(a ** 3. / (self.star.mass + planet.mass * 954.7919e-6)) * 365.
                else:
                    p = np.sqrt(a ** 3. / (self.star.mass)) * 365.

                orbit_set = Orbit(self.e,
                                  p,
                                  self.t0,
                                  self.i,
                                  self.w,
                                  self.o,
                                  self.star,
                                  planet)

                detection_map = np.zeros((len(self.data_sets),self.nsamples),dtype=int)

                for d,data in enumerate(self.data_sets):
                    #ecc_anomaly = orbit_set.solve_kepler_eqn(data.times)
                    # need to add error in that function if certain values in orbit are arrays

                    x, y, z, rv = orbit_set.get_orbit_solution(data.times,exact_total_mass=exact_total_mass)

                    if isinstance(data,ImagingDataSet):

                        proj_sep = (x.flatten()**2. + y.flatten()**2.)**0.5
                        #print(np.min(proj_sep),np.max(proj_sep))
                        rho = 0 # calc from x/y

                        phys_sep = ((x.flatten()**2. + y.flatten()**2. + z.flatten()**2.)**0.5)*self.star.dist

                        planet_contrast = self.model.get_contrast(planet, self.star, 'nominal', data.filter, phys_sep)
                        #print(data.f_contrast(proj_sep))

                        # compare to contrast of data set for proje
                        if data.ndim == 1:
                            detection_map[d,:] = planet_contrast < data.f_contrast(proj_sep)#,rho)
                        elif data.ndim == 2:
                            detection_map[d, :] = planet_contrast < data.f_contrast(x.flatten(),y.flatten())  # ,rho)
                        #print(hi)

                    elif isinstance(data,RVDataSet):

                        rv_diff = np.max(rv, axis=0) - np.min(rv, axis=0)

                        detection_map[d,:] = rv_diff > (5*data.rv_std)

                    else:
                        raise ValueError("Data sets must be either RVDataSet or ImagingDataSet.")

                    completeness_map[d,i,j] = len(np.where(detection_map[d,:])[0])/float(self.nsamples)

                    # if completeness_map[d,i,j] != 0:
                    #     print(f"(a, m_p) = ({a}, {planet.mass}): {completeness_map[d,i,j]}")

                if len(self.data_sets) > 1:
                    print(detection_map[1])
                    print()

                    # detected in at least one data set
                    completeness_map[len(self.data_sets),i,j] = len(np.where(np.sum(detection_map,axis=0) > 0)[0]) / float(self.nsamples)

                    # detected in all data sets
                    completeness_map[len(self.data_sets)+1,i,j] = len(np.where(np.sum(detection_map, axis=0) == len(self.data_sets))[0]) / float(self.nsamples)


        return completeness_map