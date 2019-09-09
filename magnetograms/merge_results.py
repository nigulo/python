import sys
sys.path.append('../utils')
import os
import misc
import numpy as np
import plot

results = None
for root, dirs, files in os.walk("."):
    for file in files:
        if file[:7] != "result_":
            continue
        (n1, n2, n3, num_x, num_y, x_no, y_no, field_y) = misc.load(file)
        if results is None:
            results = np.array((n1*n2*n3, 3))
        nx = n1//num_x
        ny = n2//num_y
        results[nx*ny*n3*x_no + ny*n3*y_no] = field_y
y = np.reshape(results, (n1, n2, n3, 3))
misc.save("results.pkl", y)

x1_range = 1.0
x2_range = 1.0
x3_range = x1_range*n3/n1

x1 = np.linspace(0, x1_range, n1)
x2 = np.linspace(0, x2_range, n2)
x3 = np.linspace(0, x3_range, n3)

x1_mesh, x2_mesh, x3_mesh = np.meshgrid(x1, x2, x3, indexing='ij')

my_plot = plot.plot(nrows=n3, ncols=1)
my_plot.set_color_map('bwr')
for layer in np.arange(0, n3):
    my_plot.colormap(y[:, :, layer, 2], ax_index = [layer])
    my_plot.vectors(x1_mesh, x2_mesh, y[:, :, layer, 0], y[:, :, layer, 0], ax_index = [layer], units='width', color = 'k')
my_plot.save("result.png")
my_plot.close()
