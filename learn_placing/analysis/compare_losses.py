import numpy as np

from learn_placing.common.vecplot import AxesPlot
from learn_placing.common.transformations import Rx, Ry, Rz
from learn_placing.training.utils import wrap_torch_fn, compute_geodesic_distance_from_two_matrices, point_loss

geodesic_loss = compute_geodesic_distance_from_two_matrices

Rbase = Rx(0.73)@Ry(0.73)
Rbasez = Rbase@Rz(1.4)

axp = AxesPlot()

Rbase = Rx(0.73)@Ry(0.73)
vbase = Rbase@[0,0,-1,1]

axp.plot_v(vbase[:3])

for th in np.linspace(0.0,np.pi,10):
    Rbasez = Rbase@Rz(th)
    vbasez = Rbasez@[0,0,-1,1]

    axp.plot_v(vbasez[:3], color="cyan")

    gl = np.squeeze(wrap_torch_fn(geodesic_loss, [Rbase], [Rbasez]))
    pl = np.squeeze(wrap_torch_fn(point_loss, [Rbase], [Rbasez]))

    print(f"geodesic: {gl:.2f} | nikloss: {pl:.2f}")


axp.title(f"Geodesic Loss vs. Point Loss")
axp.show()