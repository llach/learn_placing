import numpy as np

from .transformations import Qx, Qz, unit_vector, quaternion_multiply, quaternion_conjugate, quaternion_slerp, make_T

def cam2col(cam):
    all_colors = [
        np.array([217,  93,  57])/255,
        np.array([239, 203, 104])/255,
        np.array([180, 159, 204])/255
    ]
    if cam == "cam1":
        return all_colors[0]
    if cam == "cam2":
        return all_colors[1]
    if cam == "cam3":
        return all_colors[2]

def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return np.squeeze(a / np.expand_dims(l2, axis))

def rotate_v(v, q):
    v = list(unit_vector(v))
    v.append(0.0) # vector as pure quaternion, i.e. normalized and 4D
    return quaternion_multiply(
        quaternion_multiply(q, v),
        quaternion_conjugate(q)
    )[:3]

def qavg(quats):
    if len(quats)==1: return quats[0]

    n = len(quats)
    q = quats[0]
    for i in range(1, n):
        q = quaternion_slerp(q, quats[i], 1/n)
    return normalize(q)

def vecs2quat(u, v):
    theta = np.dot(u,v) + np.sqrt(np.sqrt(np.linalg.norm(u) * np.linalg.norm(v)))
    q = np.concatenate([np.cross(u,v), [theta]])
    return normalize(q)

def diff_quat(u, v):
    
    """
    https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another/1171995#1171995
    
    Quaternion q;
    vector a = crossproduct(v1, v2);
    q.xyz = a;
    q.w = sqrt((v1.Length ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);
    """
    a = np.cross(u, v)
    w = np.sqrt(np.linalg.norm(u)**2 * np.linalg.norm(v)**2) + np.dot(u, v)
    return normalize(np.concatenate([a, [w]], axis=0))

def qO2qdiff(qO): return vecs2quat([0,0,-1], rotate_v([1,0,0], qO)) 
def v_from_qdiff(qd): return rotate_v([0,0,-1], qd)

def cam_stats(seq):
    angles = {}
    dists = {}
    for dp in seq:
        qdiffs = [qO2qdiff(q) for q in dp["qOs"]]

        # over camera detections in per sample
        for qd, cam, dist in zip(qdiffs, dp["cameras"], dp["distances"]):
            v = v_from_qdiff(qd)
            cosa = np.dot(v, [0,0,-1])

            if cam not in angles: angles.update({cam: []})
            angles[cam].append(cosa)

            if cam not in dists: dists.update({cam: dist})

    
    stats = {}
    for cam, angs in angles.items():
        stats.update({cam: [len(angs), np.mean(angs), np.std(angs)]})

    return angles, stats, dists

def vec2polar(v):
    """
    cos(theta): angle between placing normal [0,0,-1] and v (in the -Y,Z plane)
    cos(phi)  : angle between X axis and v (in the X,Y plane)
    q         : quaternion that performs both rotations in order

    NOTE: when applying q to [0,0,-1], the resulting vector u might have a dot product != 1 (more like 0.995)
    """

    v = normalize(v)
    # since the y axis is left-pointing, we don't need to negate the sign here
    # the x axis is "swapped" to our in this representation
    cos_th = np.dot(normalize([v[1], v[2]]), [0,-1])
    cos_phi = np.dot(normalize([v[0], v[1]]), [0,1])

    q = normalize(quaternion_multiply(
            Qz(np.arccos(cos_phi)),
            Qx(np.arccos(cos_th)),
    ))
    
    return cos_th, cos_phi, q

def get_T(tfs, target, source):
    for t in tfs:
        if t["parent_frame"]==source and t["child_frame"]==target:
            return make_T(t["translation"], t["rotation"])
    return -1

def extract_gripper_T(tfs):
    world2obj = []
    grip2obj = []

    T = None
    for t in tfs:
        if len(t)>4 and T is None:
                Ts = [
                    make_T([0.000, 0.000, 0.099], [0.000, 0.000, 0.000, 1.000]), # footprint -> base
                    make_T([-0.062, 0.000, 0.193], [0.000, 0.000, 0.000, 1.000]), # base -> torso fixed
                    get_T(t, "torso_lift_link", "torso_fixed_link"),
                    get_T(t, "arm_1_link", "torso_lift_link"),
                    get_T(t, "arm_2_link", "arm_1_link"),
                    get_T(t, "arm_3_link", "arm_2_link"),
                    get_T(t, "arm_4_link", "arm_3_link"),
                    get_T(t, "arm_5_link", "arm_4_link"),
                    get_T(t, "arm_6_link", "arm_5_link"),
                    get_T(t, "arm_7_link", "arm_6_link"),
                    make_T([-0.000, 0.000, 0.077], [-0.707, 0.707, -0.000, -0.000]), # arm 7 -> gripper
                    make_T([0.000, 0.000, -0.120],  [-0.500, 0.500, 0.500, 0.500]) # gripper -> grasping frame
                ]

                T = np.eye(4)
                for TT in Ts:
                    T = T@TT
                continue
        for tr in t:
            if tr["child_frame"] == "object" and tr["parent_frame"] == "base_footprint":
                world2obj.append(tr["rotation"])
            elif tr["child_frame"] == "grasped_object" and tr["parent_frame"] == "gripper_grasping_frame":
                grip2obj.append(tr["rotation"])
        
    return T, world2obj, grip2obj
