import os
import numpy as np

from learn_placing.common.data import load_dataset_file
from learn_placing.common.vecplot import AxesPlot
from learn_placing.common.label_processing import normalize, rotate_v, vec2polar
from learn_placing.common.transformations import Rx, Rz

dsname = "second"
dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{dsname}.pkl"
ds = load_dataset_file(dataset_file_path)

vecs = np.array([d["vec"] for d in list(ds["labels"].values())])

v = vecs[0]
cos_th, cos_phi, q = vec2polar(v)

vv = np.array(list(v)+[1])
R = Rx(np.arccos(cos_th))@Rz(np.arccos(cos_phi))
u = vv@R
u = u[:-1]

# u = normalize(rotate_v([0,1,0], q))
print(cos_th)
print(cos_phi)

print(np.dot(u,v))

axp = AxesPlot()
axp.title("Polar Coordinates Calculation Verification")
axp.plot_v(v, color="grey", label="rotation label")
axp.plot_v(u, color="black", label="re-computed polar angles")
axp.show()