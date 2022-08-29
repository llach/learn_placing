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

minz = np.min(z_angles)
maxz = np.max(z_angles)
print("z", minz, maxz, np.rad2deg(minz), np.rad2deg(maxz),  np.rad2deg(maxz)-np.rad2deg(minz))

minx = np.min(x_angles)
maxx = np.max(x_angles)
print("x", minx, maxx, np.rad2deg(minx), np.rad2deg(maxx),  np.rad2deg(maxx)-np.rad2deg(minx))

z_angles = np.array([[np.cos(za), np.sin(za)] for za in z_angles])
x_angles = np.array([[np.cos(xa), np.sin(xa)] for xa in x_angles])

plt.scatter(z_angles[:,0], z_angles[:,1], color="blue", label="z axis angular difference")
plt.scatter(x_angles[:,0], x_angles[:,1], color="red", label="x axis angular difference")

plt.xlim([-1.05, 1.05])
plt.ylim([-1.05, 1.05])

plt.title("Object in Gripper Angles")
plt.xlabel("cos(theta)")
plt.ylabel("sin(theta)")

plt.legend()

plt.savefig(f"{__file__.replace(__file__.split('/')[-1], '')}/../plots/gripper_object_angle.png")
plt.show()