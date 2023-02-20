import os
import rospy
import base64
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tactile_msgs.msg import TactileState
from geometry_msgs.msg import WrenchStamped

from tf import TransformListener, TransformBroadcaster
from learn_placing.common import v2l, line_angle_from_rotation, models_theta_plot, preprocess_myrmex, tft
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
        self.li.waitForTransform(self.world_frame, self.object_frame, rospy.Time(0), rospy.Duration(5))
        
    def tl_cb(self, m): self.mm_left  = preprocess_myrmex(m.sensors[0].values)
    def tr_cb(self, m): self.mm_right = preprocess_myrmex(m.sensors[0].values)
    def ft_cb(self, m): self.ft = np.concatenate([v2l(m.wrench.force), v2l(m.wrench.torque)])

    def estimate(self):
        while np.any([self.mm_left, self.mm_right, self.ft] == None):
            print("waiting for data ...")
            rospy.Rate(1).sleep()

        # get gripper and object orientations
        try:
            # TODO make this more sensitive to lost TFs while avoiding extrapolation into the future exception
            (_, Qwg) = self.li.lookupTransform(self.world_frame, self.grasping_frame,    rospy.Time())
            (_, Qwo) = self.li.lookupTransform(self.world_frame, self.object_frame,      rospy.Time())
            (_, Qgo) = self.li.lookupTransform(self.grasping_frame, self.object_frame,   rospy.Time())
        except Exception as e:
            print(f"ERROR couldn't get transform. ¿is the object being detected?\n{e}")
            return

        Qwg = np.array(Qwg)
        Qgo = np.array(Qgo)

        # preprocess data
        mm = np.squeeze(np.stack([self.mm_left, self.mm_right]))
        lblth = line_angle_from_rotation(Qgo)

        # run models
        (R_nn, nnth), (nnRerr, nnerr) = self.nn.estimate_transform(mm, Qgo, Qwg=Qwg)
        (_, pcath), (_, pcaerr) = self.pca.estimate_transform(mm, Qgo)
        (_, houth), (_, houerr) = self.hough.estimate_transform(mm, Qgo)

        print()
        print(f"LBL {lblth:.4f}")
        print(f"NN  {nnth:.4f} | {nnerr:.4f} | {nnRerr:.4f}")
        print(f"PCA {pcath:.4f} | {pcaerr:.4f}")
        print(f"HOU {houth:.4f} | {houerr:.4f}")


        self.br.sendTransform(
            [0,0,0],
            tft.quaternion_from_matrix(tft.ensure_homog(R_nn)),
            rospy.Time.now(),
            "object_nn",
            self.grasping_frame
        )

        if self.publish_image:
            scale=100
            fig, ax = plt.subplots(ncols=1, figsize=0.8*np.array([10,9]))

            self.pca.plot_PCs(ax, mm, scale=scale)
            models_theta_plot(
                mm_imgs=mm,
                noise_thresh=self.noise_thresh,
                ax=ax,
                fig=fig,
                scale=scale,
                lines = [
                    [lblth, "target", "green"],
                    [nnth,  f"NN  {nnerr:.3f}", "red"],
                    [pcath, f"PCA {pcaerr:.3f}", "blue"],
                    [houth, f"HOU {houerr:.3f}", "white"],
                ]
            )

            ax.set_title("NN Baseline Comparison")
            fig.tight_layout()
            fig.canvas.draw()

            # Now we can save it to a numpy array.
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            imgmsg = self.bridge.cv2_to_imgmsg(data, encoding="rgb8")
            self.imgpub.publish(imgmsg)
            plt.close()

if __name__ == "__main__":
    noise_thresh = 0.05
    trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/ias_training_new_ds/Combined3D/Combined3D_Neps40_static_tactile_2022.09.13_10-41-43"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/ias_training_new_ds/Combined3D/Combined3D_Neps40_static_tactile_gripper_2022.09.13_10-42-03"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2023.02.13_18-45-21/CombinedAll/CombinedAll_Neps40_static_tactile_2023.02.13_18-45-21"

    rospy.init_node("run_estimator")


    re = RunEstimators(trial_path, noise_thresh=noise_thresh, publish_image=True)
    while not rospy.is_shutdown(): re.estimate()