import os
import pickle 
import rosbag
import numpy as np
from datetime import datetime


def time2dt(t):
    return datetime.fromtimestamp(int(str(t))/10**9)

def msg2matrix(msg):
    return np.array(msg.sensors[0].values, dtype=int)

def msg2str(msg):
    return msg.data

# base_path = f"{__file__.replace(__file__.split('/')[-1], '')}/"
base_path = f"{os.environ['HOME']}/placing_data/"
for fi in os.listdir(base_path):
    if "bag" not in fi or ".py" in fi: continue
    
    try:
        fpath = f"{base_path}/{fi}"
        bag = rosbag.Bag(fpath)
        msgs = {}
    except Exception as e:
        print(f"could not read bag file {fi}\n{e}")

    topics = []
    for topic, msg, t in bag.read_messages():
        if topic not in topics: topics.append(topic)
        if topic not in msgs: msgs.update({topic: [[], []]})

        if hasattr(msg, "header"): t = msg.header.stamp

        if "myrmex" in topic or "tactile" in topic:
            msgs[topic][1].append(msg2matrix(msg))
        elif topic == "tf":
            pass
        elif topic == "ft":
            pass
        elif topic == "object_state":
            pass
        elif topic == "joint_states":
            pass
        elif topic == "contact":
            msgs[topic][1].append(bool(msg.in_contact))
        elif topic == "bag_times":
            msgs[topic][1].append(msg2str(msg))
        else:
            continue
        # we only reach this line if the `else` above wasn't reached
        msgs[topic][0].append(time2dt(t))
    break
    with open(fpath.replace(".bag", ".pkl"), "wb") as f:
        pickle.dump(msgs, f)

pass