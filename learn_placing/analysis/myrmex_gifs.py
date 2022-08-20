import os
import numpy as np

from PIL import Image
from learn_placing.common.data import load_dataset, preprocess_myrmex, mm2img, upscale_repeat

def store_mm_sample_gif(data_left, data_right, name, store_path, M=50):
    print(f"creating {name}")
    
    # convert tactile vector to tactile matrix
    le = preprocess_myrmex(data_left)
    le = upscale_repeat(le, factor=10)
    le = mm2img(le)

    ri = preprocess_myrmex(data_right)
    ri = upscale_repeat(ri, factor=10)
    ri = mm2img(ri)

    # cut to same length (we determined that in `myrmez_lookback.py`)
    ri = ri[-M:]
    le = le[-M:]

    # we'll put a divider between the two sensor images based on image dims
    div_h = ri.shape[1]
    div_w = int(0.05*ri.shape[2])
    div = np.reshape(np.repeat([255], M*div_h*div_w*3), (M, div_h, div_w, 3))

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

for t, sample in ds.items():
    name = t.strftime("%Y-%m-%d_%H:%M:%S")

    data_left = sample["tactile_left"][1]
    data_right = sample["tactile_right"][1]

    store_mm_sample_gif(data_left, data_right, name=name, store_path=store_path)