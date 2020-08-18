import numpy as np
from scipy.interpolate import interp1d, griddata
from analysis_tools.bbflux import bbflux
import astropy.units as u
import astropy.constants as c


def mass_radius_relationship(m_p):
    """
    :param m_p: mass of planets in Earth masses
    :type m_p: np.ndarray

    :returns rad: radius of planets in Earth radii
    """

    m_p = np.asarray(m_p)
    rad = np.zeros(m_p.shape)

    m0,r0 = 124, 12.1

    low_m = np.where(m_p < m0)
    high_m = np.where(m_p >= m0)

    rad[low_m] = r0*(m_p[low_m]/m0)**0.55
    rad[high_m] = r0*(m_p[high_m]/m0)**0.01
    
    return rad


def radius_mass_relationship(r_p):
    """
    :param r_p: radius of planets in Earth radii
    :type r_p: np.ndarray

    :returns rad: radius of planets in Earth radii
    """

    r_p = np.asarray(r_p)
    m_p = np.zeros(r_p.shape)

    m0, r0 = 124, 12.1
    m_max = 1660  # max mass for planets in paper
    r_max = r0*(m_max/m0)**0.01

    low_r = np.where(r_p < r0)
    high_r = np.where( (r_p >= r0) & (r_p < r_max))
    invalid_r = np.where(r_p >= r_max)

    if len(invalid_r[0]) > 0:
        print(f"Warning: mass is not well defined for radii > {r_max:1.1f}, so a mass of {m_max} Earth masses is assumed.")

    m_p[low_r] = m0 * (r_p[low_r] / r0) ** (1/0.55)
    m_p[high_r] = m0 * (r_p[high_r] / r0) ** (1/0.01)
    m_p[invalid_r] = m_max

    return m_p


def get_sep_rho(contr_map, pixscale):

    # get center of map
    x0, y0 = (contr_map.shape[1]-1)/2.0, (contr_map.shape[0]-1)/2.0

    xcoord = np.arange(contr_map.shape[1]) - x0
    ycoord = np.arange(contr_map.shape[0]) - y0

    X, Y = np.meshgrid(xcoord, ycoord)

    sep = pixscale * (X**2.0 + Y**2.0)**0.5
    rho = (np.arctan2(Y, X) * (180. / np.pi) - 90.) % 360

    return sep, rho


def get_xy(contr_map, pixscale):
    """
    Array of x values and array of y values for each pixel in contr_map
    in units of arcseconds relative to the center.

    Assumes N is up (i.e., y increases to the top) and E is to the left (i.e., x increases to the left).

    :param contr_map:
    :param pixscale:
    :return:
    """

    # get center of map
    x0, y0 = (contr_map.shape[1]-1)/2.0, (contr_map.shape[0]-1)/2.0

    xcoord = np.flip(np.arange(contr_map.shape[1]) - x0)
    #xcoord = np.arange(contr_map.shape[1]) - x0
    ycoord = np.arange(contr_map.shape[0]) - y0

    X, Y = np.meshgrid(xcoord, ycoord)

    return pixscale*X, pixscale*Y


def calc_limitingmass_phoenix(mag_diff,star,filt,age_star,interp_age=True):
    """
    """
    
    #age_star = star.get_age(age_type)
    
    # get absolute mag of limiting contrast
    mag_p = mag_diff + star.mags[filt]
    absmag_p = mag_p - 5*np.log10(star.dist) + 5  # absolute mag
    
    if not interp_age:
        # get model grid for this age/filter
        #model_grid, t_model, filt_idx_model = get_model_grid(filt,age_star)
        mass, temp_eff, logg, model_abs, t_model = get_model_grid(filt,age_star)

        f_absmagtojupmass = interp1d(model_abs,mass/0.0009546,fill_value = (mass[-1]/0.0009546, mass[0]/0.0009546),bounds_error=False)
        f_absmagtotemp = interp1d(model_abs,temp_eff,fill_value = (temp_eff[-1], temp_eff[0]),bounds_error=False)

        mass_limit = f_absmagtojupmass(absmag_p)
        temp_eff_lim = f_absmagtotemp(absmag_p)
        
    else:
        mass, temp_eff, logg, model_abs, model_times = get_model_grid(filt)
        t_model = age_star

        mass_limit = griddata( (model_abs, model_times), mass/0.0009546, [absmag_p, age_star] )
        temp_eff_lim = griddata((model_abs,model_times), temp_eff, [absmag_p, age_star])


    belowidx = np.where(mass_limit == (mass[-1]/0.0009546))
    aboveidx = np.where(mass_limit ==  (mass[0]/0.0009546))

    return mass_limit, t_model, belowidx, aboveidx, temp_eff_lim


def calc_contrast_phoenix(m_p, star, filter, age_star, inst='NaCo', model='AMES-Cond-2000'):
    """
    Calculate the contrast of a planet as predicted by the PHOENIX models.

    :param m_p: mass of planet for which to determine contrast in Jupiter masses
    :type m_p: float
    :param star: host star
    :type star: Star instance
    :param filter: name of filter for which to determine contrast
    :type filter: str
    :param age_star: age of star in Gyr
    :type age_star: float
    :param inst: instrument
    :type inst: str
    :param model: model to use to determine contrast
    :type model: str

    :return: contrast
    :rtype: float
    """

    # get absolute mag of limiting contrast
    mass, _, _, model_abs, model_times = get_model_grid(filter, inst=inst, model=model)

    absmag_p = griddata((model_times,mass / 0.0009546), model_abs, [age_star, m_p])

    mag_p = absmag_p + 5 * np.log10(star.dist) - 5
    contrast = mag_p - star.mags[filter]

    return contrast


def calc_contrast_teq(planet_radius,
                      r,
                      star,
                      albedo=0.3,
                      extra_heat=0.1):
    """


    :param planet_radius: radius of planet in Earth radii
    :param r: physical separation of planet in AU
    :param star: Star instance
    :param albedo:
    :param extra_heat:
    :return:
    """

    # filter info
    nband_w1 = 9.8
    nband_w2 = 12.4

    # not sure if this is the most logical way to deal with the extra heat
    t_eq = (1.0 + extra_heat) * star.teff * np.sqrt(star.radius*c.R_sun.to('AU').value/(2*r)) * ((1-albedo)**0.25)

    waves = np.arange(nband_w1,nband_w2,0.1)*u.micron

    waves = waves.reshape((len(waves),1))
    t_eq = np.tile(t_eq,(len(waves),1))

    bb_star = bbflux(star.teff*u.K,waves).value
    bb_planet = bbflux(t_eq*u.K,waves).value

    bb_ratio = np.sum(bb_planet,axis=0)/np.sum(bb_star,axis=0)

    flux_ratio = bb_ratio * ((planet_radius*c.R_earth)/(star.radius*c.R_sun))**2.0

    return(-2.5*np.log10(flux_ratio.value))


def get_model_grid(filt, age_star=None, inst='NaCo',model='AMES-Cond-2000'):
    '''
    age_star (default = None): if None, then model values for ALL ages are returned.
    '''

    model_dir = '/Users/annaboehle/research/code/grids/'
    model_fname = 'model.{:s}.M-0.0.{:s}.Vega'.format(model,inst)
    
    model_file = open(model_dir + model_fname)
    lines = model_file.readlines()

    # get available times in this model file
    t_idx = []
    times = []
    for ll,line in enumerate(lines):
        if 't (Gyr)' in line:
            t_idx.append(ll)
        
            t = line.lstrip('   t (Gyr) =')
            t = t.rstrip('\n')
            times.append(float(t))

    times = np.array(times)

    # get list of column labels
    all_filts = lines[t_idx[0]:t_idx[1]][2].split()

    # fix issue that first column label is combined: M/MsTeff(K)
    all_filts.insert(0,'M/Ms')
    all_filts[1] = 'Teff(K)'
    all_filts = np.array(all_filts)
    filt_idx = np.where(all_filts == filt)[0][0]
    
    if age_star:
        # select time to use
        t_row = np.argmin(np.abs(times-age_star))

        # get model grid
        # 0:4 are headers, -4:[end] is the end -> 4:-4 is the data!
        if t_row != (len(times)-1):
            model_grid_ls = [ np.array(line.split())[[0,1,3,filt_idx]] for line in lines[t_idx[t_row]:t_idx[t_row+1]][4:-4]]
        else:
            model_grid_ls = [ np.array(line.split())[[0,1,3,filt_idx]] for line in lines[t_idx[t_row]:][4:-4]]
        model_grid = np.array(model_grid_ls,dtype='float')

       

    else:
        model_grid_ls = []
        model_times_ls = []
        
        # loop over all times
        for i in range(len(t_idx)):
            if i != (len(times)-1):
                model_grid_age_ls = [  np.array(line.split())[[0,1,3,filt_idx]] for line in lines[t_idx[i]:t_idx[i+1]][4:-4]]
            else:
                model_grid_age_ls = [  np.array(line.split())[[0,1,3,filt_idx]] for line in lines[t_idx[i]:][4:-4]]
            model_grid_ls.extend(model_grid_age_ls)
            
            model_times_ls.extend([times[i]]*len(model_grid_age_ls))

        model_grid = np.array(model_grid_ls,dtype='float')
        model_times = np.array(model_times_ls,dtype='float')

    # get model values to return
    mass = model_grid[:,0]
    temp_eff = model_grid[:,1]
    logg = model_grid[:,2]
    model_abs = model_grid[:,3]
    
    if  age_star:
        return mass, temp_eff, logg, model_abs, times[t_row]
    else:
        return mass, temp_eff, logg, model_abs, model_times
