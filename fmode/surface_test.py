from geomdl import construct
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
import numpy as np

points = []
for z in np.linspace(1, 2, 10):
    for phi in np.linspace(0, 2*np.pi, 10):
        x = z*np.cos(phi)
        y = z*np.sin(phi)
        w = x**2 + y**2
        points.append((x, y, w))


surf = fitting.approximate_surface(points, size_u=10, size_v=10, degree_u=3, degree_v=3)

# Extract curves from the approximated surface
surf_curves = construct.extract_curves(surf)
plot_extras = [
    dict(
        points=surf_curves['u'][0].evalpts,
        name="u",
        color="cyan",
        size=5
    ),
    dict(
        points=surf_curves['v'][0].evalpts,
        name="v",
        color="magenta",
        size=5
    )
]

# Plot the interpolated surface
surf.delta = 0.05
surf.vis = vis.VisSurface()
surf.render(extras=plot_extras)
