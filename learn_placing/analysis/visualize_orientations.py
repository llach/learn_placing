import os 
import numpy as np

from learn_placing.common import load_dataset, cam_stats, vecs2quat, rotate_v, cam2col
from learn_placing.common.vecplot import AxesPlot

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
    # if i not in [15, 16]: continue
    if i not in [43, 108]: continue

    angles, stats, dists = cam_stats(v[1])

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
    # else:
    #     continue

    axp = AxesPlot()
    # over sequence samples
    plotted_cams = []
    for j, dp in enumerate(v[1]):
        qdiffs = [vecs2quat([0,0,-1], rotate_v([1,0,0], q)) for q in dp["qOs"]]

        # over camera detections in per sample
        for k, qd, cam, dist in zip(range(len(qdiffs)), qdiffs, dp["cameras"], dp["distances"]):
            vec = rotate_v([0,0,-1], qd)
            cosa = np.dot(vec, [0,0,-1])
            legend = cam not in plotted_cams
            axp.plot_v(vec, color=cam2col(cam), label=f"{cam}; cos(a)={cosa:.4f}; dist={dist:.2f}; len={stats[cam][0]}; std={stats[cam][2]:.3f}", legend=legend)
            if legend: plotted_cams.append(cam)

    axp.title(f"Cuboid Dataset Sample No. {i+1}")
    axp.show()
print(sorted(set(tbc)))
pass