# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:02:01 2018

@author: olspern1
"""
import numpy as np
import numpy.random as random

def bilinear_interp(xs, ys, x, y):
    coefs = np.zeros(len(xs)*len(ys))
    h = 0
    for k in np.arange(0, len(ys)):
        for i in np.arange(0, len(xs)):
            num = 1.0
            denom = 1.0
            for l in np.arange(0, len(ys)):
                if k != l:
                    for j in np.arange(0, len(xs)):
                        if i != j:
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

def calc_W(u_mesh, us, xys, dim = 2):
    W = np.zeros((np.shape(xys)[0]*dim, np.shape(us)[0]*dim))
    i = 0
    for (x, y) in xys:
        (u1s, u2s), (indices_u1, indices_u2) = get_closest(u_mesh[0][0,:], u_mesh[1][:,0], x, y)
        coefs = bilinear_interp(u1s, u2s, x, y)
        coef_ind = 0
        for u2_index in indices_u2:
            for u1_index in indices_u1:
                j = u2_index * len(u_mesh[0]) + u1_index
                for i1 in np.arange(0, dim):
                    for j1 in np.arange(0, dim):
                        if i1 == j1:
                            W[dim*i+i1,dim*j+j1] = coefs[coef_ind]
                #W[2*i,2*j] = coefs[coef_ind]
                #W[2*i,2*j+1] = 0.0#coefs[coef_ind]
                #W[2*i+1,2*j] = 0.0#coefs[coef_ind]
                #W[2*i+1,2*j+1] = coefs[coef_ind]
                coef_ind += 1
        assert(coef_ind == len(coefs))
        i += 1
    return W
