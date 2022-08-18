import numpy as np
from data_processing import load_dataset
from tf.transformations import unit_vector, quaternion_multiply, quaternion_conjugate, quaternion_inverse, quaternion_slerp, quaternion_about_axis, quaternion_matrix, inverse_matrix, quaternion_from_matrix

def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return np.squeeze(a / np.expand_dims(l2, axis))

def rotate_v(v, q):
    v = list(unit_vector(v))
    v.append(0.0) # vector as pure quaternion, i.e. normalized and 4D
    return quaternion_multiply(
        quaternion_multiply(q, v),
        quaternion_conjugate(q)
    )[:3]

def vecs2quat(u, v):
    theta = np.dot(u,v) + np.sqrt(np.sqrt(np.linalg.norm(u) * np.linalg.norm(v)))
    q = np.concatenate([np.cross(u,v), [theta]])
    return normalize(q)

def diff_quat(u, v):
    
    """
    https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another/1171995#1171995
    
    Quaternion q;
    vector a = crossproduct(v1, v2);
    q.xyz = a;
    q.w = sqrt((v1.Length ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);
    """
    a = np.cross(u, v)
    w = np.sqrt(np.linalg.norm(u)**2 * np.linalg.norm(v)**2) + np.dot(u, v)
    return normalize(np.concatenate([a, [w]], axis=0))

dataset_path = "/home/llach/tud_datasets/2022.08.17_second/placing_data_pkl"
ds = load_dataset(dataset_path)
os = {k: v["object_state"] for k, v in ds.items()}


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
import numpy as np

class Arrow3D(FancyArrowPatch):

    def __init__(self, base, head, mutation_scale=20, lw=3, arrowstyle="-|>", color="r", **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), mutation_scale=mutation_scale, lw=lw, arrowstyle=arrowstyle, color=color, **kwargs)
        self._verts3d = list(zip(base,head))

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

all_colors = [
    np.array([217,  93,  57])/255,
    np.array([239, 203, 104])/255,
    np.array([180, 159, 204])/255
]
vecs = []

for i, (k, v) in enumerate(os.items()):
    print(i)
    fig = plt.figure(figsize=(9.71, 8.61))
    ax = fig.add_subplot(111, projection='3d')
    alim = [-1.2, 1.2]
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_zlim(alim)

    aalph = 0.9
    ax.add_artist(Arrow3D([0,0,0], [1,0,0], color=[1.0, 0.0, 0.0, aalph]))
    ax.add_artist(Arrow3D([0,0,0], [0,1,0], color=[0.0, 1.0, 0.0, aalph]))
    ax.add_artist(Arrow3D([0,0,0], [0,0,1], color=[0.0, 0.0, 1.0, aalph]))

    handles = []
    handles.append(
        ax.add_artist(Arrow3D([0,0,0], [0,0,-1], color=[0.0, 1.0, 1.0, 0.7], label="desired normal"))
    )
    for dp in v[1]:
        qdiffs = [vecs2quat([0,0,-1], rotate_v([1,0,0], q)) for q in dp["qOs"]]
        vdiffs = [list(np.array(vo)-np.array(vc)) for vc, vo in zip(dp["vcurrents"], dp["voffsets"])]
        dots = [np.dot(np.array(vo), np.array(vc)) for vc, vo in zip(dp["vcurrents"], dp["voffsets"])]
        quats = [diff_quat(vc, vo) for vc, vo in zip(dp["vcurrents"], dp["voffsets"])]

        for qd, cam, cosa in zip(qdiffs, dp["cameras"], dp["angles"]):
            v = rotate_v([0,0,-1], qd)
            handles.append(
                ax.add_artist(Arrow3D([0,0,0], v, color=[0.0, 1.0, 1.0, 1.0], label=f"{cam} normal; cos(a)={cosa:.3f}"))
            )

    # ax.legend(handles=handles)
        
    fig.tight_layout()
    fig.canvas.draw()
    plt.show()
    
    print("\n\n")
pass