import os
from learn_placing.common.tools import label_to_theta
import rospy
import numpy as np

from tactile_msgs.msg import TactileState
from geometry_msgs.msg import WrenchStamped

from tf import TransformListener, TransformBroadcaster
from learn_placing.common import v2l
from learn_placing.common.myrmex_processing import preprocess_myrmex
from learn_placing.estimators import NetEstimator, PCABaseline, HoughEstimator


class RunEstimators:
    world_frame = "base_footprint"
    grasping_frame = "gripper_grasping_frame"
    object_frame = "object"
    mm_left, mm_right, ft = None, None, None

    def __init__(self, trial_path, noise_thresh):
        self.trial_path = trial_path
        self.noise_thresh = noise_thresh
        
        # create NN and baseline models
        self.nn = NetEstimator(trial_path)
        self.pca = PCABaseline(noise_thresh=self.noise_thresh)
        self.hough = HoughEstimator(noise_thresh=self.noise_thresh, preproc="binary")

        self.ftsub = rospy.Subscriber("/wrist_ft", WrenchStamped, callback=self.ft_cb)
        self.tlsub = rospy.Subscriber("/tactile_left",  TactileState, callback=self.tl_cb)
        self.trsub = rospy.Subscriber("/tactile_right", TactileState, callback=self.tr_cb)

        self.br = TransformBroadcaster()
        self.li = TransformListener()
        self.li.waitForTransform(self.world_frame, self.object_frame, rospy.Time(0), rospy.Duration(5))
        
    def tl_cb(self, m): self.mm_left  = preprocess_myrmex(m.sensors[0].values)
    def tr_cb(self, m): self.mm_right = preprocess_myrmex(m.sensors[0].values)
    def ft_cb(self, m): self.ft = np.concatenate([v2l(m.wrench.force), v2l(m.wrench.torque)])

    def estimate(self):
        while np.any([self.mm_left, self.mm_right, self.ft] == None):
            print("waiting for data ...")
            rospy.Rate(1).sleep()

        # get gripper and object orientations
        (_, Qwg) = self.li.lookupTransform(self.world_frame, self.grasping_frame, rospy.Time(0))
        (_, Qgo) = self.li.lookupTransform(self.world_frame, self.object_frame,   rospy.Time(0))

        Qwg = np.array(Qwg)
        Qgo = np.array(Qgo)

        # preprocess data
        print(self.mm_left)
        mm = np.squeeze(np.stack([self.mm_left, self.mm_right]))
        lblth = label_to_theta(Qgo)

        # run models
        (R_nn, nnth), (nnRerr, nnerr) = self.nn.estimate_transform(mm, Qgo, Qwg=Qwg)
        (_, pcath), (_, pcaerr) = self.pca.estimate_transform(mm, Qgo)
        (_, houth), (_, houerr) = self.hough.estimate_transform(mm, Qgo)

        print()
        print(f"PCA err {pcaerr:.4f} | NN  err {nnerr:.4f} | HOU err {houerr:.4f}")
        print(f"PCA th  {pcath:.4f} | NN  th  {nnth:.4f} | HOU th  {houth:.4f} | LBL th {lblth:.4f}")

if __name__ == "__main__":
    noise_thresh = 0.15
    trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/ias_training_new_ds/Combined3D/Combined3D_Neps40_static_tactile_2022.09.13_10-41-43"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/ias_training_new_ds/Combined3D/Combined3D_Neps40_static_tactile_gripper_2022.09.13_10-42-03"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2023.02.13_18-45-21/CombinedAll/CombinedAll_Neps40_static_tactile_2023.02.13_18-45-21"

    rospy.init_node("run_estimator")


    re = RunEstimators(trial_path, noise_thresh=noise_thresh)
    re.estimate()