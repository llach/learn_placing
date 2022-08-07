import numpy as np
from PIL import Image

from data_processing import preprocess_myrmex

def upscale_repeat(frame, factor=10):
    return frame.repeat(factor, axis=1).repeat(factor, axis=2)

import pickle

base_path = __file__.replace(__file__.split('/')[-1], '')
with open(f"{base_path}/2022-07-27-16-44-18.pkl", "rb") as f:
    mm_data = pickle.load(f)

# convert tactile vector to tactile matrix
left = preprocess_myrmex(mm_data["/tactile_left"])
right = preprocess_myrmex(mm_data["/tactile_right"])

left_chan = np.zeros(list(left.shape) + [3])
left_chan[:,:,:,2] = left
left_chan *= 255
left_chan = left_chan.astype(np.uint8)
left_chan = upscale_repeat(left_chan, 35)

# right_chan = np.zeros(list(right.shape) + [3])
# right_chan[:,:,:,2] = right
# right_chan *= 255
# right_chan = right_chan.astype(np.uint8)

imgs = [Image.fromarray(img) for img in left_chan]
for i, img in enumerate(imgs):
    img.save(f"{base_path}/images/{i}.png")

# imgs[200].show()
# duration is the number of milliseconds between frames; this is 40 frames per second
# imgs[0].save(f"{base_path}/array.gif", save_all=True, append_images=imgs[1:], duration=100, loop=1)