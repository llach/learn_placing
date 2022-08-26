import os
import numpy as np

from learn_placing.common import load_dataset_file, preprocess_myrmex

name="second"
dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{name}.pkl"
ds = load_dataset_file(dataset_file_path)

inp = np.mean(np.mean(list(ds["inputs"].values()), axis=-1), axis=-1)
static_inps = np.mean(np.mean(list(ds["static_inputs"].values()), axis=-1), axis=-1)

batch=inp.shape[0]
M=inp.shape[-1]

# Nsamples*2, sequence
inp = inp.reshape((2*batch,M))
static_inps = static_inps.reshape((2*batch,M))
    
series = np.array(series)

means = np.mean(series, axis=0)
std = np.std(series, axis=0)

import matplotlib.pyplot as plt

plt.figure(figsize=(8.71, 6.61))

plt.plot(xs, means)
plt.fill_between(xs, means-std, means+std, alpha=0.4)
plt.ylabel("Average Sensor Activation: dynamic vs static")

plt.axvline(-50)
plt.title("Sensor Activation")
plt.xlabel("Sequence Time Step")
plt.tight_layout()
plt.show()