import os
import numpy as np

from learn_placing.common.data import load_dataset_file
from learn_placing.common.vecplot import AxesPlot
from learn_placing.common.label_processing import normalize, rotate_v, vec2polar

dsname = "second"
dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{dsname}.pkl"
ds = load_dataset_file(dataset_file_path)

vecs = np.array([d["vec"] for d in list(ds["labels"].values())])

v = vecs[0]
cos_th, cos_phi, q = vec2polar(v)

u = normalize(rotate_v([0,0,-1], q))
print(cos_th)
print(cos_phi)

print(np.dot(u,v))

axp = AxesPlot()
axp.title("Polar Coordinates Calculation Verification")
axp.plot_v(v, color="grey", label="rotation label")
axp.plot_v(u, color="black", label="re-computed polar angles")
axp.show()