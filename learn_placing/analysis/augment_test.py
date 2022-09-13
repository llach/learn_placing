import os
import numpy as np
import matplotlib.pyplot as plt
from learn_placing.common.myrmex_processing import mm2img, upscale_repeat, random_shift_seq

from learn_placing.training.utils import InRot, ds2name, DatasetName
from learn_placing.common.data import load_dataset_file
from learn_placing.common.transformations import quaternion_matrix

def frames2img(left_frame, right_frame):
    lm = upscale_repeat(np.expand_dims(left_frame, 0), 10)
    rm = upscale_repeat(np.expand_dims(right_frame, 0), 10)

    lm = mm2img(lm)
    rm = mm2img(rm)

    div_h = rm.shape[1]
    div_w = int(0.05*rm.shape[2])
    div = np.reshape(np.repeat([255], 1*div_h*div_w*3), (1, div_h, div_w, 3))

    # combine matrices ...
    return np.squeeze(np.concatenate([lm,div,rm], axis=2).astype(np.uint8))

dsname = DatasetName.cuboid_large
dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{ds2name[dsname]}.pkl"
ds = load_dataset_file(dataset_file_path)

labels = list(ds["labels"].values())
mms = np.array([d for d in list(ds["static_inputs"].values())])

Rgo = quaternion_matrix(labels[0][InRot.g2o])[:3,:3]
print(np.rad2deg(np.arccos(np.dot([0,0,-1], Rgo@[0,0,-1]))))

""" sequence-wise test
"""
mmax = np.max(mms[0,:])
mms /= mmax

# we take a sequence of tactile frames
seq = mms[0]
augment = [True, True] # rows, clumns
sseq = random_shift_seq(seq, augment)

mm = frames2img(seq[0,0,:], seq[1,0,:])
smm = frames2img(sseq[0,0,:], sseq[1,0,:])

fig = plt.figure(figsize=(9.71, 8.61))

ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.imshow(mm)
ax2.imshow(smm)

plt.tight_layout()
plt.show()