import numpy as np
from PIL import Image

from data_processing import normalize_mm, reshape_mm_vector

def upscale_repeat(frame, factor=10):
    return frame.repeat(2, axis=factor).repeat(2, axis=factor)


import pickle

# read data
with open(f"{__file__.replace(__file__.split('/')[-1], '')}/test_samples/2022-08-05.15:25:03.bag", "rb") as f:
    mm_data = pickle.load(f)

# convert tactile vector to tactile matrix
left = reshape_mm_vector(mm_data["/tactile_left"])
right = reshape_mm_vector(mm_data["/tactile_right"])

# one matrix for all samples and normalize
data = np.concatenate([left, right], axis=0)
data = normalize_mm(data)

imgs = np.random.randint(0, 255, (100, 50, 50, 3), dtype=np.uint8)
imgs = [Image.fromarray(img) for img in imgs]
# duration is the number of milliseconds between frames; this is 40 frames per second
imgs[0].save("array.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)