import numpy as np

class TFEstimator:

    def estimate_transform(self, mm_imgs: np.ndarray, lbl: np.ndarray, *a, **kw):
        """
        mm_imgs: [2x16x16], left & right myrmex sensor images stacked
        lbl:     [1x4], quaternion of gripper to object transform, Qgo

        returns:
        1) tf tuple
            Rgo [3x3] Rotation matrix sensor to object
            theta     line angle relative to x-axis 
        2) error tuple
            RotMat error
            theta error (line similarity)
        """
        raise NotImplementedError