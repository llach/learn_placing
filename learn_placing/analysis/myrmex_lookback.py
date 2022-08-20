import os
import numpy as np

from learn_placing.common import load_dataset, preprocess_myrmex

dataset_path = f"{os.environ['HOME']}/tud_datasets/placing_data_pkl_second"
# sample timestamp -> sample
ds = load_dataset(dataset_path)

store_path = f"{__file__.replace(__file__.split('/')[-1], '')}/test_samples"

Ns = []
M = 90
series = []
for t, sample in ds.items():
    name = t.strftime("%Y-%m-%d_%H:%M:%S")

    data_left = sample["tactile_left"][1]
    data_right = sample["tactile_right"][1]

    le = np.reshape(preprocess_myrmex(data_left), (-1, 256))
    ri = np.reshape(preprocess_myrmex(data_right), (-1, 256))

    N = min(ri.shape[0], le.shape[0])
    Ns.append(N)

    # keep only latest M samples, equalizing length
    # also bias-correct sequences by subtracting the first sample
    ri = ri[-M:]-ri[-M]
    le = le[-M:]-le[-M]

    series.append(np.mean(le, axis=1))
    series.append(np.mean(ri, axis=1))
    
series = np.array(series)

means = np.mean(series, axis=0)
std = np.std(series, axis=0)

import matplotlib.pyplot as plt

xs = np.arange(M)-M

for ys in series:
    plt.plot(xs, ys)
plt.title("mean sensor activation over sample time")

# plt.plot(xs, means)
# plt.fill_between(xs, means-std, means+std, alpha=0.4)
# plt.title("mean sensor activation over sample time (averaged over dataset)")


plt.tight_layout()
plt.show()