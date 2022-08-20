import os
import numpy as np

from PIL import Image
from data_processing import preprocess_myrmex, mm2img, upscale_repeat, load_dataset

def store_mm_sample_gif(data_left, data_right, name, store_path):

    # convert tactile vector to tactile matrix
    le = preprocess_myrmex(data_left)
    le = upscale_repeat(le, factor=10)
    le = mm2img(le)

    ri = preprocess_myrmex(data_right)
    ri = upscale_repeat(ri, factor=10)
    ri = mm2img(ri)

    # cut to whichever side had the least samples
    N = min(ri.shape[0], le.shape[0])
    ri = ri[-N:]
    le = le[-N:]

    # we'll put a divider between the two sensor images based on image dims
    div_h = ri.shape[1]
    div_w = int(0.05*ri.shape[2])
    div = np.reshape(np.repeat([255], N*div_h*div_w*3), (N, div_h, div_w, 3))

    # combine matrices ...
    mm = np.concatenate([le,div,ri], axis=2).astype(np.uint8)

    # convert to image array
    imgs = [Image.fromarray(img) for img in mm]

    # duration is the number of milliseconds between frames
    imgs[0].save(f"{store_path}/{name}.gif", save_all=True, append_images=imgs[1:], duration=100, loop=1)

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

    # break
    # store_mm_sample_gif(data_left, data_right, name=name, store_path=store_path)
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