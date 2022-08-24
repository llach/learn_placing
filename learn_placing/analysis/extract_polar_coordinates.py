import os
import numpy as np

from learn_placing.common.data import load_dataset_file
from learn_placing.common.vecplot import AxesPlot
from learn_placing.common.label_processing import normalize, rotate_v, vec2polar, vecs2quat
from learn_placing.common.transformations import Rx, Rz, Ry, euler_from_quaternion

dsname = "second"
dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{dsname}.pkl"
ds = load_dataset_file(dataset_file_path)

vecs = np.array([d["vec"] for d in list(ds["labels"].values())])

v = vecs[1]

q = vecs2quat([0,0,-1], v)

# cos_th = np.dot(normalize([v[1], v[2]]), [0,-1])
# cos_phi = np.dot(normalize([v[0], v[1]]), [0,1])

R = Rz(3*np.pi/4)@Rx(-3*np.pi/4)
v = R[:3,:3]@[0,0,-1]
v = normalize(v)

print("th", )
print("phi",)

theta = np.arctan2(v[1], -v[2])
phi = np.arctan2(v[0], -v[1])
print(theta, phi)

# w = Rx(-theta)[:3,:3]@v
# print(np.arctan2(w[1], -w[2]))
# print(np.arctan2(w[0], -w[1]))


# R = Ry(phi)#@Rx(theta)
# u = R@[1,0,0,1]
# u = u[:-1]

# tu, tp, _ = vec2polar(u)
# u = normalize(rotate_v([0,1,0], q))

print("v", v)
print("v", np.linalg.norm(v))

# print("u", u)
# print("u", np.linalg.norm(u))

axp = AxesPlot()
axp.title("Polar Coordinates Calculation Verification")
axp.plot_v(v, color="grey", label="rotation label")
# axp.plot_v(w, color="black", label="re-computed polar angles")
axp.show()

pass