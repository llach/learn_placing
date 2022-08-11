import os
import pickle 
import rosbag
import numpy as np
from datetime import datetime

def q2l(q): return [q.x, q.y, q.z, q.w]
def v2l(vec): return [vec.x, vec.y, vec.z]

def tparse(tra):
    return {
        "stamp": time2dt(tra.header.stamp), 
        "parent_frame": tra.header.frame_id, 
        "child_frame": tra.child_frame_id,
        "translation": v2l(tra.transform.translation),
        "rotation": q2l(tra.transform.rotation),
    }

def time2dt(t):
    return datetime.fromtimestamp(int(str(t))/10**9)

def msg2matrix(msg):
    return np.array(msg.sensors[0].values, dtype=int)

def msg2str(msg):
    return msg.data

def msg2js(msg):
    return msg.name, {"position": msg.position, "velocity": msg.velocity}

def msg2ft(msg):
    return [v2l(msg.wrench.force), v2l(msg.wrench.torque)]

def msg2tf(msg):
    return [tparse(t) for t in msg.transforms]

def msg2os(msg):
    return {
        "angle": msg.angle.data,
        "angles": [a.data for a in msg.angles],
        "cameras": [c.data for c in msg.cameras],
        "transform": tparse(msg.transform),
        "vcurrents": [v2l(v) for v in msg.vcurrents],
        "voffsets": [v2l(v) for v in msg.voffsets]
    }

base_path = f"/home/llach/tud_datasets/2022.08.09_first/placing_data/"
store_base = f"{base_path[:-1]}_pkl/"
os.makedirs(store_base, exist_ok=True)

with open(f"{base_path}/flagged.txt", "r") as f:
    flagged = [li.replace("\n", "") for li in f.readlines() if len(li)>5]

print(f"{len(flagged)} samples were flagged")

nsamples = 0
filtered = 0

for fi in os.listdir(base_path):
    if "bag" not in fi or ".py" in fi: continue
    sample_name = fi.replace(".bag", "")

    # don't process flagged sample
    if sample_name in flagged:
        print(f"skipping flagged sample {sample_name}")
        continue

    # there was a bug where the file name was empty (1 out of 200), so we skip these
    if len(sample_name) < 10: 
        print("empty file name. skipping")
        continue
    
    try:
        fpath = f"{base_path}/{fi}"
        store_path = f"{store_base}/{fi.replace('.bag', '.pkl')}"
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
            msgs[topic][1].append(msg2tf(msg))
        elif topic == "ft":
            msgs[topic][1].append(msg2ft(msg))
        elif topic == "object_state":
            msgs[topic][1].append(msg2os(msg))
            pass
        elif topic == "joint_states":
            joint_names, m = msg2js(msg)
            if "joint_names" not in msgs: msgs.update({"joint_names": joint_names})
            msgs[topic][1].append(m)
        elif topic == "contact":
            msgs[topic][1].append(bool(msg.in_contact))
        elif topic == "bag_times":
            msgs[topic][1].append(msg2str(msg))
        else:
            continue
        # we only reach this line if the `else` above wasn't reached

        msgs[topic][0].append(time2dt(t))
    
    # sanity check
    for k, v in msgs.items():
        if k == "joint_names": continue
        assert len(v[0])==len(v[1]), f"length mismatch for {k}"

    if "object_state" not in topics:
        print("no object state")
        filtered += 1
        continue 

    with open(store_path, "wb") as f:
        pickle.dump(msgs, f)
    nsamples += 1

print(f"{nsamples} converted; {len(flagged)} flagged; {filtered} filtered; total {len(flagged)+nsamples+filtered}")
pass