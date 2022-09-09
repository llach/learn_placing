import os
import numpy as np

from learn_placing.common import load_dataset

# dataset_path = f"{os.environ['HOME']}/tud_datasets/placing_data_pkl_seven"
dataset_path = f"{os.environ['HOME']}/placing_data_pkl"
# sample timestamp -> sample
ds = load_dataset(dataset_path)

this_path = __file__.replace(__file__.split('/')[-1], '')
plot_path = f"{this_path}/../plots/"
# store_path = f"{__file__.replace(__file__.split('/')[-1], '')}/test_samples"

Ns = []
L = 30
series = []
for t, sample in ds.items():
    name = t.strftime("%Y-%m-%d_%H:%M:%S")
    data_ft = np.reshape(sample["ft"][1][-L:], (L,6))
    data_ft -= data_ft[0]

    series.append(data_ft)

series = np.mean(series, axis=2)

means = np.mean(series, axis=0)
std = np.std(series, axis=0)

import matplotlib.pyplot as plt

plt.figure(figsize=(8.71, 6.61))
xs = np.arange(L)-L

for ys in series:
    plt.plot(xs, ys)
plt.ylabel("Average mean FT")

# plt.plot(xs, means)
# plt.fill_between(xs, means-std, means+std, alpha=0.4)
# plt.ylabel("Average Sensor Activation in Dataset")

plt.axvline(-15)
plt.title("Sensor Activation")
plt.xlabel("time steps until sequence end")
plt.tight_layout()
plt.show()