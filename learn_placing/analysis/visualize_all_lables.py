import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from learn_placing.training.utils import InRot
from learn_placing.common.data import load_dataset_file
from learn_placing.common.vecplot import AxesPlot
from learn_placing.common.transformations import quaternion_matrix

dsname = "cylinder_large"
dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{dsname}.pkl"
ds = load_dataset_file(dataset_file_path)

vecs = np.array([quaternion_matrix(d[InRot.w2o])[:3,:3]@[0,0,-1] for d in list(ds["labels"].values())])

fig = plt.figure(figsize=(9.71, 4.61))
gs = gridspec.GridSpec(1,2)

axp = AxesPlot(fig=fig, sps=gs[0,0], angles=(-90,90))
axp.plot_points(vecs, color="grey", label="rotation samples")
axp.axtitle("Object Rotations (X-Y Plane)")

axp2 = AxesPlot(fig=fig, sps=gs[0,1], angles=(-90,0))
axp2.plot_points(vecs, color="grey", label="rotation samples")
axp2.axtitle("Object Rotations (X-Z Plane)")

axp.show()