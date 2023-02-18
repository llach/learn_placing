import os
import cv2
import numpy as np

from learn_placing.common.data import load_dataset
from learn_placing.analysis.baseline_trials import extract_sample, label_to_theta, merge_mm_samples

import matplotlib.pyplot as plt

from learn_placing.common.myrmex_processing import upscale_repeat, mm2img



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
    mmm = merge_mm_samples(mm, noise_tresh=0.15)
    mmmimg = mm2img(mmm)
    
    lblth = label_to_theta(lbl)

    # Convert the image to gray-scale
    grey = cv2.cvtColor(mmmimg, cv2.COLOR_BGR2GRAY)
    # Find the edges in the image using canny detector
    # edges = cv2.Canny(grey, 20, 50)
    edges = np.where(mmm>0.1, 1, 0).astype(np.uint8)
    # cv2.imshow("", upscale_repeat(edges, factor=100))
    # cv2.waitKey()
    # cv2.imshow("", upscale_repeat(grey, factor=100))
    # cv2.waitKey()

    # Detect points that form a line
    lines = cv2.HoughLines(edges,1,np.pi/180,7)
    print(len(lines))


    mmiscaled = mm2img(upscale_repeat(mmm, factor=100))
    # mmiscolor = cv2.cvtColor(mmiscaled, cv2.COLOR_GRAY2BGR)

    rho,theta = lines[0][0] # we take the line with the maximum votes
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a*rho*100
    y0 = b*rho*100
    x1 = int(x0 + 5000*(-b))
    y1 = int(y0 + 5000*(a))
    x2 = int(x0 - 5000*(-b))
    y2 = int(y0 - 5000*(a))
    print((x1,y1),(x2,y2))
    print(np.pi/2-theta)
    cv2.line(mmiscaled,(x1,y1),(x2,y2),(255,0,255),2)

    cv2.imshow("", mmiscaled)
    cv2.waitKey()

    # Draw lines on the image
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    # # Show result
    # cv2.imshow("Result Image", img)