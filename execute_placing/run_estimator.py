import os
import rospy
import numpy as np
import matplotlib.pyplot as plt

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tactile_msgs.msg import TactileState
from geometry_msgs.msg import WrenchStamped

from tf import TransformListener, TransformBroadcaster
from datetime import datetime
from learn_placing.common import v2l, line_angle_from_rotation, models_theta_plot, preprocess_myrmex, tft, rotation_from_line_angle
from learn_placing.estimators import NetEstimator, PCABaseline, HoughEstimator


class RunEstimators:
    world_frame = "base_footprint"
    grasping_frame = "gripper_grasping_frame"
    object_frame = "object"
    mm_left, mm_right, ft = None, None, None

    def __init__(self, trial_path, noise_thresh, publish_image=False):
        self.trial_path = trial_path
        self.noise_thresh = noise_thresh
        self.publish_image = publish_image
        
        # create NN and baseline models
        self.nn = NetEstimator(trial_path)
        self.pca = PCABaseline(noise_thresh=self.noise_thresh)
        self.hough = HoughEstimator(noise_thresh=self.noise_thresh, preproc="binary")

        self.ftsub = rospy.Subscriber("/wrist_ft", WrenchStamped, callback=self.ft_cb)
        self.tlsub = rospy.Subscriber("/tactile_left",  TactileState, callback=self.tl_cb)
        self.trsub = rospy.Subscriber("/tactile_right", TactileState, callback=self.tr_cb)

        if self.publish_image: 
            self.bridge = CvBridge()
            self.imgpub = rospy.Publisher("/estimator_image", Image, queue_size=1)

        self.br = TransformBroadcaster()
        self.li = TransformListener()
        self.li.waitForTransform(self.world_frame, self.grasping_frame, rospy.Time(0), rospy.Duration(5))
        
    def tl_cb(self, m): self.mm_left  = preprocess_myrmex(m.sensors[0].values)
    def tr_cb(self, m): self.mm_right = preprocess_myrmex(m.sensors[0].values)
    def ft_cb(self, m): self.ft = np.concatenate([v2l(m.wrench.force), v2l(m.wrench.torque)])

    def reset_data(self):
        self.mm_left, self.mm_right, self.ft = None, None, None

    def estimate(self):
        while np.any([self.mm_left, self.mm_right, self.ft] == None):
            print("waiting for data ...")
            rospy.Rate(1).sleep()

        # get gripper and object orientations
        (_, Qwg) = self.li.lookupTransform(self.world_frame, self.grasping_frame,    rospy.Time())
        try:
            # TODO make this more sensitive to lost TFs while avoiding extrapolation into the future exception
            (_, Qwo) = self.li.lookupTransform(self.world_frame, self.object_frame,      rospy.Time())
            (_, Qgo) = self.li.lookupTransform(self.grasping_frame, self.object_frame,   rospy.Time())
            detected = True
        except Exception as e:
            print(f"ERROR couldn't get transform. ¿is the object being detected?\n{e}")
            Qgo = [0,0,0,1]
            detected = False

        Qwg = np.array(Qwg)
        Qgo = np.array(Qgo)

        # preprocess data
        mm = np.squeeze(np.stack([self.mm_left, self.mm_right]))
        lblth = line_angle_from_rotation(Qgo)

        # run models
        (R_nn, nnth), (nnerr, _) = self.nn.estimate_transform(mm, Qgo, Qwg=Qwg, ft=[self.ft.copy()])
        (R_pca, pcath), (pcaerr, _) = self.pca.estimate_transform(mm, Qgo)
        (R_hou, houth), (houerr, _) = self.hough.estimate_transform(mm, Qgo)

        print()
        print(f"LBL {lblth:.4f}")
        print(f"NN  {nnth:.4f} | {nnerr:.4f}")
        print(f"PCA {pcath:.4f} | {pcaerr:.4f}")
        print(f"HOU {houth:.4f} | {houerr:.4f}")

        # broadcast transforms
        for name, R in zip(["nn", "pca", "hough"], [R_nn, R_pca, R_hou]):
            if R is None or np.any(np.isnan(np.array(R))): # handle NaNs / models not detecting lines
                print(f"skipping {name} due to NaN")
                continue

            T = tft.ensure_homog(R)
            self.br.sendTransform(
                [0,0,0],
                tft.quaternion_from_matrix(T),
                rospy.Time.now(),
                f"object_{name}",
                self.grasping_frame
            )

        if self.publish_image:
            scale=100
            fig, ax = plt.subplots(ncols=1, figsize=0.8*np.array([10,9]))

            self.pca.plot_PCs(ax, mm, scale=scale)

            if detected:
                lines = [
                    [lblth, f"OptiTrack (lblth)", "green"],
                    [nnth,  f"NN  {nnerr:.3f}", "red"],
                    [pcath, f"PCA {pcaerr:.3f}", "blue"],
                    [houth, f"HOU {houerr:.3f}", "white"],
                ]
            else:
                lines = [
                    [nnth,  f"NN ", "red"],
                    [pcath, f"PCA", "blue"],
                    [houth, f"HOU", "white"],
                ]

            models_theta_plot(
                mm_imgs=mm,
                noise_thresh=self.noise_thresh,
                ax=ax,
                fig=fig,
                scale=scale,
                lines=lines
            )

            ax.set_title(f"Estimation Results [{datetime.now().strftime('%H:%M:%S')}]")
            fig.tight_layout()
            fig.canvas.draw()

            # Now we can save it to a numpy array.
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            imgmsg = self.bridge.cv2_to_imgmsg(data, encoding="rgb8")
            self.imgpub.publish(imgmsg)
            plt.close()

        result = dict(zip(["nn", "pca", "hough"],
            [
                [tft.ensure_homog(R_nn), nnerr],
                [R_pca, pcaerr],
                [R_hou, houerr],
            ]
        ))
        if detected:
            result.update({"opti": [rotation_from_line_angle(lblth), 0]})
        return result

if __name__ == "__main__":
    noise_thresh = 0.05
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/ias_training_new_ds/Combined3D/Combined3D_Neps40_static_tactile_2022.09.13_10-41-43"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/ias_training_new_ds/Combined3D/Combined3D_Neps40_static_tactile_gripper_2022.09.13_10-42-03"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2023.02.13_18-45-21/CombinedAll/CombinedAll_Neps40_static_tactile_2023.02.13_18-45-21"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2023.02.22_15-25-54/UPC_v1/UPC_v1_Neps60_static_tactile_2023.02.22_15-25-54"

    # trial_path = f"{os.environ['HOME']}/tud_datasets/chosen_ones/UPC_v1_Neps60_static_tactile_2023.02.23_09-27-55"
    trial_path = f"{os.environ['HOME']}/tud_datasets/chosen_ones/UPC_v1_Neps60_static_tactile_ft_2023.02.23_14-04-41"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/chosen_ones/UPC_v1_Neps60_static_ft_2023.02.23_14-04-25"

    rospy.init_node("run_estimator")

    # normal streaming
    re = RunEstimators(trial_path, noise_thresh=noise_thresh, publish_image=True)
    while not rospy.is_shutdown(): re.estimate()

    # re = RunEstimators(trial_path, noise_thresh=noise_thresh, publish_image=True)
    # while not rospy.is_shutdown():
    #     a = input()
    #     if a.lower() == "q": break

    #     re.estimate()
