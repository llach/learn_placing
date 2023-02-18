import os
import rospy

from tf import TransformBroadcaster
from learn_placing.common.data import load_dataset_file
from learn_placing.training.train_utils import InRot

rospy.init_node("w2o_debug")
br = TransformBroadcaster()

name = "cuboid_large"
dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{name}.pkl"
ds = load_dataset_file(dataset_file_path)
Y =  [d[InRot.w2o] for d in list(ds["labels"].values())]


i = 0
qcurr = None
r = rospy.Rate(20)
interval = rospy.Duration(secs=1)
last_pub_time = rospy.Time.now()-interval
while not rospy.is_shutdown():
    if last_pub_time+interval < rospy.Time.now():
        if i==len(Y): break
        print(f"broadcasting Y[{i}]")
        qcurr = Y[i]
        
        i += 1
        last_pub_time = rospy.Time.now()

    br.sendTransform(
        [0,0,0],
        qcurr,
        rospy.Time.now(),
        "object_ds",
        "base_footprint"
    )
    r.sleep()
print("bye")