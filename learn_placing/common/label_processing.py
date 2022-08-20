import numpy as np

from .transformations import unit_vector, quaternion_multiply, quaternion_conjugate, quaternion_slerp

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

            if cam not in angles: angles |= {cam: []}
            angles[cam].append(cosa)

            if cam not in dists: dists |= {cam: dist}

    
    stats = {}
    for cam, angs in angles.items():
        stats |= {cam: [len(angs), np.mean(angs), np.std(angs)]}

    return angles, stats, dists

