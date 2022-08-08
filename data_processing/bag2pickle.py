import os
import pickle 
import rosbag
import numpy as np


def time2dt(msg):
    return

def msg2matrix(msg):
    return np.array(msg.sensors[0].values, dtype=int)

base_path = f"{__file__.replace(__file__.split('/')[-1], '')}/"
for fi in os.listdir(base_path):
    if "bag" not in fi or ".py" in fi: continue
    
    try:
        fpath = f"{base_path}/{fi}"
        bag = rosbag.Bag(fpath)
        msgs = {}
    except Exception as e:
        print(f"could not read bag file {fi}\n{e}")

    for topic, msg, t in bag.read_messages():
        if topic not in msgs: msgs.update({topic: []})
        if "myrmex" in topic or "tactile" in topic:
            msgs[topic].append(msg2matrix(msg))
        else:
            continue

    with open(fpath.replace(".bag", ".pkl"), "wb") as f:
        pickle.dump(msgs, f)

pass