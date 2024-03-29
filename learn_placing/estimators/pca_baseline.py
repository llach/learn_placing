import numpy as np

from learn_placing.common.tools import tft, get_cov, marginal_mean, marginal_sd, line_angle_from_rotation, line_similarity, rotation_from_line_angle, to_tensors
from learn_placing.common.myrmex_processing import merge_mm_samples
from learn_placing.estimators import TFEstimator
from learn_placing.training.utils import LossType, get_loss_fn

class PCABaseline(TFEstimator):

    def __init__(self, noise_thresh) -> None:
        self.noise_thresh = noise_thresh
        self.crit = get_loss_fn(LossType.pointarccos)

    def get_PC_point(self, means, evl, evec, pci):
        return means-2*np.sqrt(evl[pci])*evec[:,pci]

    def get_line_angle(self, p1, p2):
        if p1[0]>p2[0]: p1, p2 = p2, p1
        th = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
        return th if th>0 else np.pi+th

    def calculate_PCA(self, samples):
        """
        """
        sample = merge_mm_samples(samples, noise_tresh=self.noise_thresh)
        meanx, meany = marginal_mean(sample, axis=0), marginal_mean(sample, axis=1)
        sdx, sdy = marginal_sd(sample, axis=0), marginal_sd(sample, axis=1)
        cov = get_cov(sample)
        
        C = np.array([[sdx**2, cov],
                    [cov,    sdy**2]])

        evl, evec = np.linalg.eig(C)
        eigsort = np.argsort(evl)[::-1]
        evl, evec = evl[eigsort], evec[:,eigsort]

        # slope of eigenvectors as angle theta
        # axes in mpl are flipped, hence the PI offset
        means = [meanx, meany]
        evth = np.array([
            np.pi-self.get_line_angle(means, self.get_PC_point(means, evl, evec, 0)),
            np.pi-self.get_line_angle(means, self.get_PC_point(means, evl, evec, 1)),
        ])

        return np.array([meanx, meany]), evl, evec, evth
    
    def plot_PCs(self, ax, samples, scale=1):
        """ 
        scale: upscaling factor for imshow
        """
        means, evl, evec, _ = self.calculate_PCA(samples)

        means = means.copy()
        means *= scale

        ax.text(*means,"X",color="cyan")
        ax.annotate("",
                    fontsize=20,
                    xytext = means,
                    xy     = self.get_PC_point(means, evl, evec, 0),
                    arrowprops = {"arrowstyle":"<->",
                                "color":"magenta",
                                "linewidth":2},
                    )
        ax.annotate("",
                    fontsize=20,
                    xytext = means,
                    xy     = self.get_PC_point(means, evl, evec, 1),
                    arrowprops = {"arrowstyle":"<->",
                                "color":"green",
                                "linewidth":2}
                    )

    def estimate_transform(self, mm_imgs: np.ndarray, lbl, *a, **kw):
        means, evl, evec, evth = self.calculate_PCA(mm_imgs)
        pcath = evth[0]

        lbl = tft.ensure_rotmat(lbl)
        lblth = line_angle_from_rotation(lbl)

        R_pca = np.expand_dims(rotation_from_line_angle(pcath)[:3,:3], 0)

        R_lbl = np.expand_dims(lbl, 0)
        errR = float(self.crit(*to_tensors(R_pca, R_lbl)).numpy())

        return (np.squeeze(R_pca), pcath), (errR, line_similarity(pcath, lblth))