import torch 
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import learn_placing.common.transformations as tf

from learn_placing.training.utils import qloss
from learn_placing.common.vecplot import AxesPlot
from learn_placing.common.label_processing import vecs2quat

def sample_random_orientation():
    theta, phi = 2*np.pi*np.random.uniform(0, 1), np.arccos(np.random.uniform(-1, 1))

    Qzt = tf.Qz(theta)
    Qyp = tf.Qy(phi)

    return tf.quaternion_matrix(tf.quaternion_multiply(Qzt, Qyp))

points = np.array([sample_random_orientation()[:3,:3]@[0,0,-1] for _ in range(1000)])

v = [0,-1,0]

def qloss_np(q1,q2):
    return 1-np.dot(q1,q2)**2

# vector dot product
# metric = np.array([np.dot(u,v) for u in points])
# norm_metric = 1-(metric+1)/2

# quaternion loss
qsample = vecs2quat([0,0,-1], v)
norm_metric = np.array([qloss_np(qsample, vecs2quat([0,0,-1], u)) for u in points])

# quaternion loss torch version
# norm_metric = np.squeeze(qloss(
#     torch.Tensor([vecs2quat([0,0,-1], u) for u in points]),
#     torch.Tensor(np.repeat([qsample], points.shape[0], axis=0))
# ).numpy())


cm = plt.get_cmap("copper")
colors = cm(norm_metric)

threshold = 1.0
colors = colors[np.where(norm_metric<threshold)]
points = points[np.where(norm_metric<threshold)]

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
