import numpy as np

def cart_to_polar(us):
    scalar = False
    if len(np.shape(us)) == 1:
        scalar = True
        us = np.array([us])
    rhos = np.sqrt(np.sum(us**2, axis=1))
    phis = np.arctan2(us[:,1], us[:,0])
    ret_val = np.column_stack((rhos, phis))
    if scalar:
        ret_val = ret_val[0]
    return ret_val

def polar_to_cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def trunc(ds, perc):
    p1 = np.percentile(ds, perc*100)
    p2 = np.percentile(ds, (1.0-perc)*100)
    #print(perc*100, (1.0-perc)*100, p1, p2)
    ds_out = np.array(ds)
    if perc < 0.5:
        p1a = p1
        p1 = p2
        p2 = p1a
    for i in np.arange(0, np.shape(ds_out)[0]):
        for j in np.arange(np.shape(ds_out)[1]):
            if ds_out[i, j] > p1:
                ds_out[i, j] = p1
            elif ds_out[i, j] < p2:
                ds_out[i, j] = p2
    return ds_out


