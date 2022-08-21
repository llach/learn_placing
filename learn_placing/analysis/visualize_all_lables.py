import os
import numpy as np

from learn_placing.common.data import load_dataset_file
from learn_placing.common.vecplot import AxesPlot

dsname = "second"
dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{dsname}.pkl"
ds = load_dataset_file(dataset_file_path)

vecs = np.array([d["vec"] for d in list(ds["labels"].values())])

axp = AxesPlot()
axp.plot_points(vecs, color="grey", label="rotation samples")
axp.title("Object Rotation Samples")
axp.show()