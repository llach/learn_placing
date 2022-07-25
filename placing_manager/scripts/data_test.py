import rosbag

bag_path = "/home/llach/placing_data/2022-07-22.16:04:49.bag"

def bag_to_dict(b):
    d = {}
    for topic, msg, t in b.read_messages():
        if topic not in d: d.update({topic: []})
        d[topic].append((msg, t))

    print("read bag:")
    for k, v in d.items():
        print(f"  {k} - {len(v)} msgs")
    return d

bag = rosbag.Bag(bag_path, "r")
bag_dict = bag_to_dict(bag)

pass