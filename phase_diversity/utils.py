import numpy as np

def cart_to_polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def polar_to_cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def trunc(ds, perc):
    p1 = np.percentile(ds, perc*100)
    p2 = np.percentile(ds, (1.0-perc)*100)
    print(perc*100, (1.0-perc)*100, p1, p2)
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