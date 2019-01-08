import numpy as np
import radvel
from scipy.optimize import newton, fsolve
import matplotlib.pyplot as plt

def newtons_method(t, P, t0, e):

    E_arr = np.repeat(E, len(t))
    print E_arr
    f_E = lambda E, P, t0, e: E - e*np.sin(E) - (2*np.pi/P)*(t - t0)
    
def eccentric_anamoly(t, P, t0, e, test=True):

    out = np.zeros(len(t))
    for i,time in enumerate(t):
        f_E = lambda E, P, t0, e: E - e*np.sin(E) - (2*np.pi/P)*(time - t0)        
    
        out[i]=newton(f_E, 0.0, args=(P, t0, e))
        #out=fsolve(f_E, 0.0, args=(P, t0, e))

        if test:
            E_arr = np.arange(-np.pi, np.pi, 0.1)
            l = plt.plot(E_arr,f_E(E_arr, P, t0, e))
            plt.axvline(out[i], color=l[0].get_color())

    if test:
        plt.axhline(0.0, color='black', linestyle='dotted')
        plt.show()

    print out        

def r_v_f(f, a, e):

    num = a*(1-e**2.)
    denom = 1 + e*np.cos(f)

    return num/denom

def ang_sep(f, a, e, w, i):

    r = r_v_f(f, a, e)

    del_thera = (r/d)*(np.cos)
    
    #dist_star, w, i

    f = radvel.orbit.true_anomaly(t, t0, p, e)
