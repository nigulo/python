import sys
sys.path.append('../utils')
import os
import misc
import numpy as np
import plot
import cov_div_free


subsample = 1000000
num_subsample_reps = 1
num_tries_without_progress = 100
approx = True

sig_var=1.
length_scale=.1*3
noise_var=0.01

results = None
#n1 = 30
#n2 = 30
#n3 = 3
for root, dirs, files in os.walk("."):
    for file in files:
        if file[:7] != "result_":
            continue
        (n1_, n2_, n3_, num_x_, num_y_, x_no, y_no, field_y) = misc.load(file)
        if results is None:
            n1 = n1_
            n2 = n2_
            n3 = n3_
            num_x = num_x_
            num_y = num_y_
            results = np.zeros((n1*n2*n3, 3))
        else:
            assert(n1 == n1_ and n2 == n2_ and n3 == n3_)
            assert(num_x == num_x_ and num_y == num_y_)
        assert(x_no >= 0 and x_no < num_x)
        assert(y_no >= 0 and y_no < num_x)
        nx = n1//num_x
        ny = n2//num_y
        #index = nx*ny*n3*x_no + n1*ny*n3*y_no
        print(num_x, num_y, x_no, y_no)
        index = nx*n2*n3*x_no + ny*n3*y_no
        for i in np.arange(0, nx):
            print(nx, ny, results[index+n2*n3*i:index+n2*n3*i+ny*n3].shape, field_y[ny*n3*i:ny*n3*(i+1)].shape)
            results[index+n2*n3*i:index+n2*n3*i+ny*n3] = field_y[ny*n3*i:ny*n3*(i+1)]
y_grid = np.reshape(results, (n1, n2, n3, 3))
misc.save("results.pkl", y_grid)


x1_range = 1.0
x2_range = 1.0
x3_range = x1_range*n3/n1

x1 = np.linspace(0, x1_range, n1)
x2 = np.linspace(0, x2_range, n2)
x3 = np.linspace(0, x3_range, n3)

x1_mesh, x2_mesh, x3_mesh = np.meshgrid(x1, x2, x3, indexing='ij')
x_grid = np.stack((x1_mesh, x2_mesh, x3_mesh), axis=3)
x = x_grid.reshape(-1, 3)
#x_flat = np.reshape(x, (3*n, -1))
n = n1*n2*n3

bx = np.reshape(y_grid[:, :, :, 0], n)
by = np.reshape(y_grid[:, :, :, 1], n)
bz = np.reshape(y_grid[:, :, :, 2], n)

y = np.column_stack((bx, by, bz))

ri = 0
rj = 0

def invert_random_patch(y):
    y_grid = np.reshape(y, (n1, n2, n3, 3))
    if np.random.choice([True, False]):
        i = np.random.randint(0, high = num_x)
        j = np.random.randint(0, high = num_y)
    else:
        i = ri
        j = rj
    nx = n1//num_x
    ny = n2//num_y
    x_start = i*nx
    y_start = j*ny
    x_end = min(x_start + nx, n1)
    y_end = min(y_start + ny, n2)
    y_patch = y_grid[x_start:x_end, y_start:y_end, :, :]
    y_patch[:, :, :, :2] *= -1
    
def calc_loglik(y):
    gp = cov_div_free.cov_div_free(sig_var, length_scale, noise_var)
    if (approx):
        loglik = 0.
        for i in np.arange(0, num_subsample_reps):
            loglik += gp.loglik_approx(x, np.reshape(y, (3*n, -1)), subsample=subsample)
            #if (best_loglik is None or loglik > best_loglik):
            #    best_loglik = loglik
        return loglik/num_subsample_reps
    else:
        return gp.loglik(x, np.reshape(y, (3*n, -1)))
    
###############################################################################
# Now try gradually flipping the horizontal directions of the
# consecutive patches to obtain the optimal configuration
max_loglik = None
num_tries = 1

changed = True
while max_loglik is None or num_tries % num_tries_without_progress != 0:
    y_copy = np.array(y)
    count = np.random.randint(0, num_x*num_y//2)
    for _ in np.arange(0, count):
        invert_random_patch(y)
        ri += 1
        rj += 1
        ri %= num_x
        rj %= num_y

    loglik = calc_loglik(y)

    print("loglik=", loglik, "max_loglik=", max_loglik)

    if max_loglik is None or loglik > max_loglik:
        max_loglik = loglik
        num_tries = 1
    else:
        y = y_copy
        num_tries += 1

    results_plot = plot.plot(nrows=n3, ncols=3)
    results_plot.set_color_map('bwr')
    y_grid = np.reshape(y, (n1, n2, n3, 3))
    for layer in np.arange(0, n3):
        results_plot.colormap(y_grid[:, :, layer, 0], [layer, 0])
        results_plot.colormap(y_grid[:, :, layer, 1], [layer, 1])
        results_plot.colormap(np.reshape(np.arctan2(y_grid[:, :, layer, 1], y_grid[:, :, layer, 0]), (n1, n2)), [layer, 2])
    results_plot.save("results.png")
    results_plot.close()

    misc.save("results.pkl", y_grid)

