import os
import numpy as np
import matplotlib.pyplot as plt

from learn_placing.common import load_dataset_file
from learn_placing.training.utils import InRot

name="second"
dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{name}.pkl"
ds = load_dataset_file(dataset_file_path)

z_angles = [d[InRot.gripper_angle] for d in list(ds["labels"].values())]
x_angles = [d[InRot.gripper_angle_x] for d in list(ds["labels"].values())]

z_angles = np.array([[np.sin(za), np.cos(za)] for za in z_angles])
x_angles = np.array([[np.sin(xa), np.cos(xa)] for xa in x_angles])

plt.scatter(z_angles[:,0], z_angles[:,1], color="blue")
plt.scatter(x_angles[:,0], x_angles[:,1], color="red")
plt.show()