import os
import cv2
import numpy as np

from learn_placing.common.data import load_dataset
from learn_placing.analysis.pca_trials import extract_sample, label_to_theta

import matplotlib.pyplot as plt



if __name__ == "__main__":
    """ NOTE interesting samples

    Dataset: placing_data_pkl_cuboid_large
    good: [64, 69]
    50/50: [58]
    hard: [188]

    NN  good: 100
    PCA good: 64
    PCA bad : 188
    """
    
    dsname = "placing_data_pkl_cuboid_large"
    dataset_path = f"{os.environ['HOME']}/tud_datasets/{dsname}"

    # sample timestamp -> sample
    ds = load_dataset(dataset_path)
    ds = list(ds.items())

    """
    dataset:
    timestamp - sample (i.e. dict of time series)
        |-> tactile_left
            |-> [timestamps]
            |-> [myrmex samples]
    """
    
    # load sample 
    frame_no  = 10
    sample_no = 64
    
    sample = ds[sample_no][1]
    mm, w2g, ft, lbl = extract_sample(sample)
    lblth = label_to_theta(lbl)

    # Convert the image to gray-scale
    # gray = cv2.cvtColor(simg, cv2.COLOR_BGR2GRAY)
    # Find the edges in the image using canny detector
    # edges = cv2.Canny(gray, 50, 200)
    cv2.imshow("", mm[0])
    cv2.waitKey()

    # Detect points that form a line
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
    # print(lines)

    # Draw lines on the image
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    # # Show result
    # cv2.imshow("Result Image", img)