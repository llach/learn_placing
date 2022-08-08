import pickle
import numpy as np

from PIL import Image
from data_processing import preprocess_myrmex, mm2img, upscale_repeat

def store_gif(data, side, name, base_path):

    # convert tactile vector to tactile matrix
    mm = preprocess_myrmex(data)
    mm = upscale_repeat(mm, factor=10)
    mm_imgs = mm2img(mm)

    imgs = [Image.fromarray(img) for img in mm_imgs]
    # duration is the number of milliseconds between frames
    imgs[0].save(f"{base_path}/{name}_{side}.gif", save_all=True, append_images=imgs[1:], duration=100, loop=1)


file_name = "touch_test"
base_path = __file__.replace(__file__.split('/')[-1], '')
with open(f"{base_path}/{file_name}.pkl", "rb") as f:
    mm_data = pickle.load(f)

store_gif(mm_data["/tactile_left"], "left", file_name, base_path)
store_gif(mm_data["/tactile_right"], "right", file_name, base_path)