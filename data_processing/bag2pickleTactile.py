import pickle 
import rosbag
import numpy as np

fpath = "/home/llach/robot_mitm_ws/2022-07-27-16-44-18.bag"
bag = rosbag.Bag(fpath)
msgs = {}

def msg2matrix(msg):
    return np.array(msg.sensors[0].values, dtype=int)

for topic, msg, t in bag.read_messages():
    if topic not in msgs: msgs.update({topic: []})
    msgs[topic].append(msg2matrix(msg))

with open(fpath.replace(".bag", ".pkl"), "wb") as f:
    pickle.dump(msgs, f)

with open(fpath.replace(".bag", ".pkl"), "rb") as f:
    data = pickle.load(f)

pass