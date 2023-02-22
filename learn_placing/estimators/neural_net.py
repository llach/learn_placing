import torch
import numpy as np

from learn_placing.common import tft
from learn_placing.training.utils import load_train_params, get_loss_fn
from learn_placing.training.tactile_insertion_rl import TactilePlacingNet
from learn_placing.common.tools import line_angle_from_rotation, line_similarity, to_tensors

from .tf_estimator import TFEstimator

class NetEstimator(TFEstimator):
    
    def __init__(self, trial_path) -> None:
        self.trial_path = trial_path
        self.trial_weights = f"{self.trial_path}/weights/best.pth"
    
        self.params = load_train_params(self.trial_path)
        self.model = TactilePlacingNet(**self.params.netp)
        self.crit = get_loss_fn(self.params.loss_type)

        self.model.load_state_dict(torch.load(self.trial_weights))
        self.model.eval()

    def estimate_transform(self, mm_imgs: np.ndarray, lbl, Qwg, *a, **kw):
        if type(lbl)==torch.Tensor: lbl = lbl.numpy()
        if lbl.shape == (3,3): lbl = tft.quaternion_from_matrix(lbl)

        # predict rotation
        pred = self.model(*to_tensors(np.expand_dims(mm_imgs, 0), np.expand_dims(Qwg, 0), 0))
        Rgo = np.squeeze(pred.detach().numpy())

        # calculate line theta
        lblth = line_angle_from_rotation(lbl)
        theta  = line_angle_from_rotation(Rgo)
        thetaerr = line_similarity(theta, lblth)

        # prepare data for loss calculation
        Rgo = np.expand_dims(Rgo, 0)
        R_lbl = np.expand_dims(tft.quaternion_matrix(lbl), 0)
        return (Rgo, theta), (float(self.crit(*to_tensors(Rgo, R_lbl)).numpy()), thetaerr)
