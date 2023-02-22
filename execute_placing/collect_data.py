import os
import time
import rospy
import pickle

import numpy as np
import matplotlib.pyplot as plt

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tactile_msgs.msg import TactileState
from geometry_msgs.msg import WrenchStamped

from tf import TransformListener
from datetime import datetime
from learn_placing import dataset_path, datefmt
from learn_placing.common import v2l, line_angle_from_rotation, models_theta_plot, preprocess_myrmex


class DataCollector:
    world_frame = "base_footprint"
    grasping_frame = "gripper_grasping_frame"
    object_frame = "object"
    mm_left, mm_right, ft = None, None, None

    def __init__(self, save_path):
        self.count = 0
        self.save_path = save_path
        self.pics_path = f"{self.save_path}pics/"

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.pics_path, exist_ok=True)

        self.ftsub = rospy.Subscriber("/wrist_ft", WrenchStamped, callback=self.ft_cb)
        self.tlsub = rospy.Subscriber("/tactile_left",  TactileState, callback=self.tl_cb)
        self.trsub = rospy.Subscriber("/tactile_right", TactileState, callback=self.tr_cb)

        self.bridge = CvBridge()
        self.imgpub = rospy.Publisher("/collector_image", Image, queue_size=1)

        self.li = TransformListener()
        self.li.waitForTransform(self.world_frame, self.grasping_frame, rospy.Time(0), rospy.Duration(5))
        
    def tl_cb(self, m): self.mm_left  = preprocess_myrmex(m.sensors[0].values)
    def tr_cb(self, m): self.mm_right = preprocess_myrmex(m.sensors[0].values)
    def ft_cb(self, m): self.ft = np.concatenate([v2l(m.wrench.force), v2l(m.wrench.torque)])

    def reset_data(self):
        self.mm_left, self.mm_right, self.ft = None, None, None

    def collect(self):
        print(f"collecting sample {self.count}")
        while np.any(self.mm_left == None) or np.any(self.mm_right == None) or np.any(self.ft == None):
            print("waiting for data ...")
            rospy.Rate(2).sleep()

        # get gripper and object orientations
        (_, Qwg) = self.li.lookupTransform(self.world_frame, self.grasping_frame,    rospy.Time())
        try:
            # TODO make this more sensitive to lost TFs while avoiding extrapolation into the future exception
            (_, Qwo) = self.li.lookupTransform(self.world_frame, self.object_frame,      rospy.Time())
            (_, Qgo) = self.li.lookupTransform(self.grasping_frame, self.object_frame,   rospy.Time())
        except Exception as e:
            print(f"ERROR couldn't get transform. ¿is the object being detected?\n{e}")
            return False

        now = datetime.now().strftime(datefmt)

        Qwg = np.array(Qwg)
        Qgo = np.array(Qgo)
        Qwo = np.array(Qwo)

        # preprocess data
        mm = np.squeeze(np.stack([self.mm_left.copy(), self.mm_right.copy()]))
        lblth = line_angle_from_rotation(Qgo)

        sname = f"{self.count}_{now}"
        with open(f"{self.save_path}{sname}.pkl", "wb") as f:
            pickle.dump({
                "mm": mm,
                "ft": self.ft,
                "Qwg": Qwg,
                "Qgo": Qgo,
                "Qwo": Qwo
            }, f)

        scale=100
        fig, ax = plt.subplots(ncols=1, figsize=0.8*np.array([10,9]))

        lines = [
            [lblth, f"OptiTrack (lblth)", "green"],
        ]
        models_theta_plot(
            mm_imgs=mm,
            noise_thresh=0.0,
            ax=ax,
            fig=fig,
            scale=scale,
            lines=lines
        )

        ax.set_title(f"sample {self.count}@{now}")
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(f"{self.pics_path}{sname}.png")

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        imgmsg = self.bridge.cv2_to_imgmsg(data, encoding="rgb8")
        self.imgpub.publish(imgmsg)
        plt.close()

        self.reset_data()
        self.count += 1
        print("done.")

      

if __name__ == "__main__":

    rospy.init_node("colelct_data")
    save_path = f"{dataset_path}upc_cuboid/"
    dc = DataCollector(save_path)

    print(f"saving data in {save_path}")
    while not rospy.is_shutdown():
        a = input()
        if a.lower() == "q": break
        for _ in range(5):
            dc.collect()
            time.sleep(0.3)
