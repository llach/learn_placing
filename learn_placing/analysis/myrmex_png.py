import os
import numpy as np

from PIL import Image
from learn_placing.common.data import load_dataset
from learn_placing.common.myrmex_processing import preprocess_myrmex, mm2img, upscale_repeat


def create_mm_png(data_left, data_right, name, store_path, M=50, preprocess=True):
    # NOTE this code was originally intended to create GIFs, and so the first dimension holds many consecutive frames from one sample
    # ideally, we should just choose one frame and process it to save on computation
    # however, I didn't have time to adapt all functions to handle a potentially missing first dimension in the case of N=1
    print(f"creating {name}")
    
    # convert tactile vector to tactile matrix
    le = preprocess_myrmex(data_left) if preprocess else data_left
    le = upscale_repeat(le, factor=10)
    le = mm2img(le)

    ri = preprocess_myrmex(data_right) if preprocess else data_right
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

    # here we just choose frame number 10
    return mm[10,:]
    

if __name__ == "__main__":
    dsname = "placing_data_pkl_cuboid_large"
    dataset_path = f"{os.environ['HOME']}/tud_datasets/{dsname}"
    # sample timestamp -> sample
    ds = load_dataset(dataset_path)

    store_path = f"{__file__.replace(__file__.split('/')[-1], '')}/test_samples"

    for i, (t, sample) in enumerate(ds.items()):
        name = t.strftime("%Y-%m-%d_%H:%M:%S")

        data_left = sample["tactile_left"][1]
        data_right = sample["tactile_right"][1]

        arr = create_mm_png(data_left, data_right, name=name, store_path=store_path, M=100)

        im = Image.fromarray(arr)
        im.save(f"/tmp/{dsname}_{i}.png")