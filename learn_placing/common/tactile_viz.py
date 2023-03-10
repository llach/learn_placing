import rospy
import numpy as np
import matplotlib.pyplot as plt

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tactile_msgs.msg import TactileState
from learn_placing.common import preprocess_myrmex, merge_mm_samples, upscale_repeat


class TactileHeatmapPublisher:
    mm_left, mm_right = None, None

    def __init__(self):
        self.tlsub = rospy.Subscriber("/tactile_left",  TactileState, callback=self.tl_cb)
        self.trsub = rospy.Subscriber("/tactile_right", TactileState, callback=self.tr_cb)

        self.bridge = CvBridge()
        self.imgpub = rospy.Publisher("/tactile_heatmap", Image, queue_size=1)

    def tl_cb(self, m): self.mm_left  = preprocess_myrmex(m.sensors[0].values)
    def tr_cb(self, m): self.mm_right = preprocess_myrmex(m.sensors[0].values)
    def reset_data(self): self.mm_left, self.mm_right = None, None

    def publish(self):
        while np.any(np.any([m == None for m in [self.mm_left, self.mm_right]])):
            print("waiting for data ...")
            rospy.Rate(1).sleep()

        # preprocess data
        mm = np.squeeze(np.stack([self.mm_left, self.mm_right]))

        fig, ax = plt.subplots(ncols=1, figsize=1.8*np.array([10,9]))

        fig = plt.figure(frameon=False)
        fig.set_size_inches(5,5)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        mmm = merge_mm_samples(mm, noise_tresh=0.0)
        mmimg = upscale_repeat(mmm, factor=100)

        ax.imshow(mmimg, aspect='auto', cmap="magma")
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        imgmsg = self.bridge.cv2_to_imgmsg(data, encoding="rgb8")
        self.imgpub.publish(imgmsg)
        plt.close("all")

if __name__ == "__main__":
    rospy.init_node("tactile_heatmap_viz")

    r = rospy.Rate(50)
    thp = TactileHeatmapPublisher()
    while not rospy.is_shutdown(): 
        thp.publish()
        r.sleep()
