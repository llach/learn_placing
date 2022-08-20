import os 
import numpy as np

from learn_placing.common import load_dataset

"""
samples second dataset:
clusters = [15, 16, 27, 28, 44, 66, 69, 80, 95, 107]
bad = [43, 108]
"""

dataset_path = f"{os.environ['HOME']}/tud_datasets/placing_data_pkl_second"
ds = load_dataset(dataset_path)
os = {k: v["object_state"] for k, v in ds.items()}

import numpy as np

vecs = []

should_plot = True
max_dev = 0.005
tbc = []
unclean = [5, 7, 15, 16, 27, 28, 43, 44, 66, 69, 71, 80, 95, 108, 116, 121, 132, 134, 135, 144, 153, 154, 159, 163, 164, 168, 173, 174, 194, 195, 200, 201]

for i, (k, v) in enumerate(os.items()):
    # print(i)
    # if i not in unclean: continue
    # if i not in [15, 16, 27, 28, 44, 66, 69, 80, 95, 107, 43, 108]: continue
    # if i not in [15, 16, 27, 28, 44, 66, 69, 80, 95, 107]: continue
    # if i not in [43, 108]: continue
    angles, stats = angle_dict(v[1])

    ignored_cams = []
    for cam, sts in stats.items():
        if sts[0] < 10: # require at least 9 samples per cam
            ignored_cams.append(cam)
        elif sts[-1] > max_dev:
            ignored_cams.append(cam)
    if ignored_cams != []: tbc.append(i)
    # if len(ignored_cams) != len(angles): continue
    if len(ignored_cams) == len(angles): 
        print(f"bad sample {i}")
    else:
        continue

    if should_plot:
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
            ax.add_artist(Arrow3D([0,0,0], [0,0,-1], color=[0.0, 1.0, 1.0, 0.7], label=f"desired normal; {i}"))
        )

    # over sequence samples
    plotted_cams = []
    for j, dp in enumerate(v[1]):
        qdiffs = [vecs2quat([0,0,-1], rotate_v([1,0,0], q)) for q in dp["qOs"]]

        # over camera detections in per sample
        for k, qd, cam, dist in zip(range(len(qdiffs)), qdiffs, dp["cameras"], dp["distances"]):
            vec = rotate_v([0,0,-1], qd)
            cosa = np.dot(vec, [0,0,-1])
            if should_plot:# and cam not in ignored_cams:
                h = ax.add_artist(Arrow3D([0,0,0], vec, color=list(cam2col(cam))+[1.0], label=f"{cam}; cos(a)={stats[cam][1]:.3f}; dist={dist:.3f}; dev={stats[cam][-1]:.5f};len={stats[cam][0]}"))
                if cam not in plotted_cams: 
                    handles.append(h)
                    plotted_cams.append(cam)

    if should_plot:
        ax.legend(handles=handles)
            
        fig.tight_layout()
        fig.canvas.draw()
        plt.show()
print(sorted(set(tbc)))
pass