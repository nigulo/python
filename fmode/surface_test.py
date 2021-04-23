from geomdl import construct
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
import numpy as np

points = []
for z in np.linspace(1, 2, 10):
    for phi in np.linspace(0, 2*np.pi, 17+np.random.randint(6)):
        phi += max(-.001, min(.001, np.random.normal()*.01))
        x = z*np.cos(phi)
        y = z*np.sin(phi)
        w = x**2 + y**2 + max(-.001, min(.001, np.random.normal()*.01))
        points.append((x, y, w))

size = int(np.sqrt(len(points)))
surf = fitting.approximate_surface(points, size_u=size, size_v=size, degree_u=2, degree_v=2)

surf.delta = 0.1
surf.evaluate(start=[0.0, 0.0], stop=[1.0, 1.0])
print(surf.evalpts)

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
surf.vis = vis.VisSurface(config=vis.VisConfig(ctrlpts=False, trims=False))
surf.render()
