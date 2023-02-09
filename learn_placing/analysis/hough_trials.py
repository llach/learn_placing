import os
import cv2
import numpy as np

from PIL import Image
from learn_placing.common.data import load_dataset
from learn_placing.common.myrmex_processing import preprocess_myrmex, mm2img, upscale_repeat

import matplotlib.pyplot as plt




if __name__ == "__main__":
    """ NOTE interesting samples

    Dataset: placing_data_pkl_cuboid_large
    good: [64, 69]
    50/50: [58]
    hard: [188]
    """
    
    dsname = "placing_data_pkl_cuboid_large"
    dataset_path = f"{os.environ['HOME']}/tud_datasets/{dsname}"

    # sample timestamp -> sample
    ds = load_dataset(dataset_path)
    samples = list(ds.items())

    sample = samples[188][1]["tactile_left"][1]
    sample = preprocess_myrmex(sample) #   16x16 array

    # sample = upscale_repeat(sample, factor=10)
    # sample = mm2img(sample)
    s = (sample[10,:] > 0.1).astype(np.uint8)
    simg = mm2img([sample[10,:]])[0,:]

    # Convert the image to gray-scale
    gray = cv2.cvtColor(simg, cv2.COLOR_BGR2GRAY)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray, 50, 200)
    cv2.imshow("", s)
    cv2.waitKey()

    # Detect points that form a line
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
    print(lines)

    # Draw lines on the image
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    # # Show result
    # cv2.imshow("Result Image", img)