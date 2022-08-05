import numpy as np
from PIL import Image

from data_processing import normalize_mm, reshape_mm_vector

def upscale_repeat(frame, factor=10):
    return frame.repeat(factor, axis=1).repeat(factor, axis=2)


import pickle

# read data
with open(f"{__file__.replace(__file__.split('/')[-1], '')}/test_samples/2022-08-05.15:25:03.pkl", "rb") as f:
    mm_data = pickle.load(f)

# convert tactile vector to tactile matrix
left = normalize_mm(reshape_mm_vector(mm_data["myrmex_left"]))
right = normalize_mm(reshape_mm_vector(mm_data["myrmex_right"]))

left_chan = np.zeros(list(left.shape) + [3])
left_chan[:,:,:,2] = left
left_chan *= 255
left_chan = left_chan.astype(np.uint8)

right_chan = np.zeros(list(right.shape) + [3])
right_chan[:,:,:,2] = right
right_chan *= 255
right_chan = right_chan.astype(np.uint8)

imgs = [Image.fromarray(img) for img in left_chan]
import matplotlib.pyplot as plt

plt.imshow(left_chan[100])
plt.show()

imgs[100].show()
# duration is the number of milliseconds between frames; this is 40 frames per second
imgs[0].save("array.gif", save_all=True, append_images=imgs[1:], duration=100, loop=1)