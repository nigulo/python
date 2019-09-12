import numpy as np
#import numpy.random as random
import itertools
import copy

'''
# Old method supporting 2 dimensions
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
                            print("points A", [x, y], [xs[i], ys[k]], [xs[j], ys[l]])
                            num *= (x - xs[j])*(y - ys[l])
                            denom *= (xs[i] - xs[j])*(ys[k] - ys[l])
            coefs[h] = num/denom
            h += 1
    return coefs
'''

'''
    Bilinear interpolation coefficients for a point in a given grid
    grid - the list with dim elements e.g. [xs, ys, zs], each element representing the 
        coordinate array in one dimension
    point - the point with shape (dim), e.g. array([x, y, z]) for which the interpolation weights are calculated
'''
def bilinear_interp(grid, point):
    size = 1
    for xs in grid:
        size *= len(xs)
    coefs = np.zeros(size)
    h = 0
    # TODO: get rid of reversings
    grid = list(reversed(grid))
    grid_points = np.array([x for x in itertools.product(*grid)]) # Cartesian product
    point = point[::-1]
    #grid_points = misc.cartesian_product(grid)
    for grid_point1 in grid_points:
        assert(point.shape[0] == grid_point1.shape[0])
        num = 1.
        denom = 1.
        for grid_point2 in grid_points:
            if np.all(grid_point1 != grid_point2):
                for i in np.arange(0, len(point)):
                    num *= (point[i] - grid_point2[i])
                    denom *= (grid_point1[i] - grid_point2[i])
        coefs[h] = num/denom
        h += 1
    return coefs

'''
# Old method supporting 2 dimensions
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
'''

'''
    Returns the arrays of coordinates of the mesh of data points
    encompassing the given data point.
    
    grid - the list with dim elements e.g. [xs, ys, zs], each element representing the 
        coordinate array in one dimension
    point - the point with shape (dim), e.g. array([x, y, z]) for which the interpolation weights are calculated
    counts - counts of mesh points to return in all directions. 
    The default is 2.    
'''
def get_closest(grid, point, counts_in = None):
    dists = []
    indices = []
    if counts_in is None:
        counts = []
    for i in np.arange(len(point)):
        dists.append(np.abs(grid[i] - point[i]))
        indices.append(np.argsort(dists[i]))
        if counts_in is None:
            counts.append(2)
    closest_coords = []
    closest_indices = []
    for i in np.arange(len(dists)):
        closest = np.zeros(counts[i])
        for j in np.arange(counts[i]):
            closest[j] = grid[i][indices[i][j]]
        closest_coords.append(closest)
        closest_indices.append(indices[i][:counts[i]])
    return closest_coords, closest_indices


'''
    Caclulates the matrix projecting one grid (usually denser)
    to another grid (usually less dense)
    coords - dense grid (n, dim)
    xs - sparse grid (m1, m2, dim)
    us - same as u_mesh, but flat (m1*m2, dim)
    dim - the dimensionality of the vector field on the grid. Default
        equals to the dimensionality of the space
'''
def calc_W(u_mesh, xs, us=None, dim = None, indexing_type=True):
    if us is None:
        us = np.dstack(u_mesh).reshape(-1, len(u_mesh))
    if dim is None:
        dim = len(u_mesh)
    W = np.zeros((np.shape(xs)[0]*dim, np.shape(us)[0]*dim))
    
    u_mesh1 = []
    indices = []
    for i in np.arange(len(u_mesh)):
        indices.append(0)
    
    for i in np.arange(len(u_mesh)):
        indices1 = copy.deepcopy(indices)
        if indexing_type:
            indices1[len(u_mesh)-i-1] = np.arange(u_mesh[i].shape[i])
        else:
            indices1[i] = np.arange(u_mesh[i].shape[i])
        u_mesh1.append(u_mesh[i][tuple(indices1)])

    i = 0
    for point in xs:
        #(u1s, u2s), (indices_u1, indices_u2) = get_closest([u_mesh[0][0,:], u_mesh[1][:,0]], point)
        closest_us, closest_inds = get_closest(u_mesh1, point)
        coefs = bilinear_interp(closest_us, point)
        coef_ind = 0
        
        closest_inds = list(reversed(closest_inds))
        closest_inds = np.array([x for x in itertools.product(*closest_inds)])
        for closest_index in closest_inds:
            j = closest_index[0]
            for iii in np.arange(1, len(closest_index)):
                j *= len(u_mesh[iii])
                j += closest_index[iii]
            for i1 in np.arange(dim):
                for j1 in np.arange(dim):
                    if i1 == j1:
                        W[dim*i+i1,dim*j+j1] = coefs[coef_ind]
            coef_ind += 1
        
        #(u1s, u2s) = closest_us
        #(indices_u1, indices_u2) = closest_inds
        #(u1s, u2s), (indices_u1, indices_u2) = get_closest(u_mesh[0][0,:], u_mesh[1][:,0], x, y)
        #coefs = bilinear_interp([u1s, u2s], np.array([x, y]))
        #for u2_index in indices_u2:
        #    for u1_index in indices_u1:
        #        j = u2_index * len(u_mesh[0]) + u1_index
        #        for i1 in np.arange(0, dim):
        #            for j1 in np.arange(0, dim):
        #                if i1 == j1:
        #                    W[dim*i+i1,dim*j+j1] = coefs[coef_ind]
        #        coef_ind += 1
        assert(coef_ind == len(coefs))
        i += 1
    return W

'''
# Old method supporting 2 dimensions
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
'''

