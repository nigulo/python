# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:02:01 2018

@author: olspern1
"""
import numpy as np

def bilinear_interp(xs, ys, x, y):
    coefs = np.zeros(len(xs)*len(ys))
    h = 0
    for i in np.arange(0, len(xs)):
        for k in np.arange(0, len(ys)):
            num = 1.0
            denom = 1.0
            for j in np.arange(0, len(xs)):
                if i != j:
                    for l in np.arange(0, len(ys)):
                        if k != l:
                            num *= (x - xs[j])*(y - ys[l])
                            denom *= (xs[i] - xs[j])*(ys[k] - ys[l])
            coefs[h] = num/denom
            h += 1
    return coefs

def get_closest(xs, ys, x, y, count_x=2, count_y=2):
    dists_x = np.abs(xs - x)
    dists_y = np.abs(ys - y)
    indices_x = np.argsort(dists_x)
    indices_y = np.argsort(dists_y)
    xs_c = np.zeros(count_x)
    ys_c = np.zeros(count_y)
    for i in np.arange(0, count_x):
        xs_c[i] = xs[indices_x[i]]
    for i in np.arange(0, count_y):
        ys_c[i] = ys[indices_y[i]]
    return (xs_c, ys_c), (indices_x[:count_x], indices_y[:count_y])

def calc_W(u_mesh, us, xys):
    W = np.zeros((np.shape(xys)[0]*2, np.shape(us)[0]*2))
    i = 0
    for (x, y) in xys:
        (u1s, u2s), (indices_u1, indices_u2) = get_closest(u_mesh[0][0,:], u_mesh[1][:,0], x, y)
        coefs = bilinear_interp(u1s, u2s, x, y)
        coef_ind = 0
        for u1_index in indices_u1:
            for u2_index in indices_u2:
                j = u2_index * len(u_mesh[0]) + u1_index
                W[2*i,2*j] = coefs[coef_ind]
                W[2*i,2*j+1] = coefs[coef_ind]
                W[2*i+1,2*j] = coefs[coef_ind]
                W[2*i+1,2*j+1] = coefs[coef_ind]
                coef_ind += 1
        assert(coef_ind == len(coefs))
        i += 1
    return W
