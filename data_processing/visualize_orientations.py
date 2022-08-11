import numpy as np
from data_processing import load_dataset
from tf.transformations import quaternion_conjugate

def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return np.squeeze(a / np.expand_dims(l2, axis))

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

dataset_path = "/home/llach/tud_datasets/2022.08.09_first/placing_data_pkl"
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


vecs = []
for k, v in os.items():
    for dp in v[1]:
        if len(dp["vcurrents"])!=2: continue
        vdiffs = [list(np.array(vo)-np.array(vc)) for vc, vo in zip(dp["vcurrents"], dp["voffsets"])]
        dots = [np.dot(np.array(vo), np.array(vc)) for vc, vo in zip(dp["vcurrents"], dp["voffsets"])]
        quats = [diff_quat(vc, vo) for vc, vo in zip(dp["vcurrents"], dp["voffsets"])]
        # print("C", dp["vcurrents"])
        # print("O", dp["voffsets"])
        # print("D", vdiffs)
        # print("Q", quats)
        # print("A", dp["angles"])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.add_artist(Arrow3D([0,0,0], dp["vcurrents"][0], color=(0.0,0.5,0.5,0.5)))
        ax.add_artist(Arrow3D([0,0,0], dp["vcurrents"][1], color=(0.0,0.0,0.5,0.5)))

        ax.add_artist(Arrow3D([0,0,0], dp["voffsets"][0], color=(0.0,0.5,0.5,1.0)))
        ax.add_artist(Arrow3D([0,0,0], dp["voffsets"][1], color=(0.0,0.0,0.5,1.0)))

        ax.add_artist(Arrow3D([0,0,0], vdiffs[0], color=(0.5,0.5,0.5,1.0)))
        ax.add_artist(Arrow3D([0,0,0], vdiffs[1], color=(0.5,0.0,0.5,1.0)))

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.show()
        break

    print("\n\n")
pass