import cv2
import numpy as np

from learn_placing.estimators import TFEstimator
from learn_placing.common.tools import tft, line_angle_from_rotation, line_similarity, ensure_positive_angle, rotation_from_line_angle, to_tensors
from learn_placing.common.myrmex_processing import merge_mm_samples, mm2img, upscale_repeat
from learn_placing.training.utils import LossType, get_loss_fn

class HoughEstimator(TFEstimator):
    
    def __init__(self, noise_thresh, preproc="canny") -> None:
        self.noise_thresh = noise_thresh
        self.preproc = preproc
        self.crit = get_loss_fn(LossType.pointarccos)

        assert self.preproc in ["canny", "binary"]

    def show_line_image(self, mmm, rho, theta):
        """
        draw detected line on merged myrmex image
        """
        mmiscaled = mm2img(upscale_repeat(mmm, factor=100))
        
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*rho*100
        y0 = b*rho*100
        x1 = int(x0 + 5000*(-b))
        y1 = int(y0 + 5000*(a))
        x2 = int(x0 - 5000*(-b))
        y2 = int(y0 - 5000*(a))

        cv2.line(mmiscaled,(x1,y1),(x2,y2),(255,0,255),2)
        cv2.imshow("Hough Line Detection on Merged Myrmex Image", mmiscaled)
        cv2.waitKey()

    def estimate_transform(self, mm_imgs: np.ndarray, lbl, show_image=False, *a, **kw):
        # merge images
        mmm = merge_mm_samples(mm_imgs, noise_tresh=self.noise_thresh)

        if self.preproc == "canny": # variant 1: canny edge detection on RGB image
            mmmimg = mm2img(mmm)
            grey = cv2.cvtColor(mmmimg, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(grey, 20, 50)
        elif self.preproc == "binary": # variant 2: binarization of merged myrmex image after filtering out noise
            edges = np.where(mmm>self.noise_thresh, 1, 0).astype(np.uint8)

        # extract lines with Hough line detection
        lines = cv2.HoughLines(edges,1,np.pi/180,7)

        """
        hough lines are parametrized by the angle of a vector normal to the line that intercepts the origin, relative to the x axis and a length of that vector
        we take the first line in the list which will be the one with the highest number of votes (alas the most confident estimate)
        if there's no line, use default values
        """
        if lines is None: 
            print("WARN no hough line found")
            return (None, np.nan), (np.pi, np.nan)
            
        rho, theta = lines[0][0]
        houth = ensure_positive_angle(np.pi/2-theta) # convert angle of line normal to angle between line and x axis

        # calculate line error
        lblth = line_angle_from_rotation(lbl)
        houerr = line_similarity(houth, lblth)

        if show_image: self.show_line_image(mmm, rho, theta)

        R_hou = np.expand_dims(rotation_from_line_angle(houth)[:3,:3], 0)
        R_lbl = np.expand_dims(tft.quaternion_matrix(lbl)[:3,:3], 0)
        errR = float(self.crit(*to_tensors(R_hou, R_lbl)).numpy())
        return (rotation_from_line_angle(houth), houth), (errR, houerr)

if __name__ == "__main__":
    import os

    from learn_placing.common.data import load_dataset
    from learn_placing.analysis.baseline_trials import extract_sample

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
    
    # load sample 
    frame_no  = 10
    sample_no = 64
    
    sample = ds[sample_no][1]
    mm, w2g, ft, lbl = extract_sample(sample)

    hough = HoughEstimator(noise_thresh=0.15, preproc="canny")
    (_, houth), (_, houerr) = hough.estimate_transform(mm, lbl, show_image=True)
    print(houth, houerr)