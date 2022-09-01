import torch 
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import learn_placing.common.transformations as tf

from learn_placing.training.utils import compute_geodesic_distance_from_two_matrices, compute_rotation_matrix_from_quaternion, point_loss, wrap_torch_fn
from learn_placing.common.vecplot import AxesPlot
from learn_placing.common.label_processing import vecs2quat

def sample_random_orientation():
    theta, phi = 2*np.pi*np.random.uniform(0, 1), np.arccos(np.random.uniform(-1, 1))

    Qzt = tf.Qz(theta)
    Qyp = tf.Qy(phi)

    return tf.quaternion_matrix(tf.quaternion_multiply(Qzt, Qyp))

N=5000
Rgoal = tf.Rx(-np.pi/2)
Rs = [tf.Rx(th)@Rgoal for th in np.linspace(0, 2*np.pi, 50)]
points = [tf.Rz(0.73)@R@[0,0,-1,1] for R in Rs]
points = np.array([p[:3] for p in points])

points = np.array([sample_random_orientation()[:3,:3]@[0,0,-1] for _ in range(N)])
# points = np.array([p for p in points if p[1]<0])

qs = np.array([vecs2quat([0,0,-1], p) for p in points])
Rs = np.array([tf.quaternion_matrix(q) for q in qs])

v = Rgoal@[0,0,-1,1]
v = v[:3]

def qloss_np(q1,q2):
    return 1-np.dot(q1,q2)**2

# vector dot product
# metric = np.array([np.dot(u,v) for u in points])
# norm_metric = 1-(metric+1)/2

# quaternion loss
# qsample = vecs2quat([0,0,-1], v)
# norm_metric = np.array([qloss_np(qsample, vecs2quat([0,0,-1], u)) for u in points])

# geodesic matrix loss
# qsample = np.repeat([vecs2quat([0,0,-1], v)], N, axis=0)
# Rsample = wrap_torch_fn(compute_rotation_matrix_from_quaternion, qsample)
# Rout = wrap_torch_fn(
#     compute_rotation_matrix_from_quaternion, 
#     np.array([vecs2quat([0,0,-1], u) for u in points])
# )

# metric = wrap_torch_fn(compute_geodesic_distance_from_two_matrices, Rsample, Rout)
# norm_metric = metric / np.pi

# point loss
def nikloss(Rtar, Rpred):
    vtar = np.array([(R@[0,0,-1,1])[:3] for R in Rtar])
    vpred = np.array([(R@[0,0,-1,1])[:3] for R in Rpred])
    coss = np.array([np.dot(u,w) for u,w in zip(vtar, vpred)])
    # return np.arccos(coss)
    return 1-coss

Rgoals =  np.repeat([Rgoal], Rs.shape[0], axis=0)

nikloss(Rgoals, Rs)

# metric = wrap_torch_fn(point_loss, Rgoals, Rs)
# norm_metric = metric / np.pi

metric = nikloss(Rgoals, Rs)
# norm_metric = metric/np.pi
norm_metric = metric/2

cm = plt.get_cmap("copper")
colors = cm(norm_metric)

# threshold = 100.0
# colors = colors[np.where(metric<threshold)]
# points = points[np.where(metric<threshold)]

axp = AxesPlot()
axp.plot_points(points, c=colors)
axp.plot_v(v, color="grey", label="sample")

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
cmappable = ScalarMappable(norm=Normalize(0,1), cmap="copper")

axp.title("loss relative to sample")
axp.fig.colorbar(cmappable, ax=axp.ax)
axp.show()

pass
