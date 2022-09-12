import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from learn_placing.common.myrmex_processing import mm2img, upscale_repeat

from learn_placing.training.utils import InRot, ds2name, DatasetName
from learn_placing.common.data import load_dataset_file
from learn_placing.common.vecplot import AxesPlot
from learn_placing.common.transformations import quaternion_matrix

def ds_analysis_plot(dsname, savepath):
    dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{ds2name[dsname]}.pkl"
    ds = load_dataset_file(dataset_file_path)

    vecs = np.array([quaternion_matrix(d[InRot.w2o])[:3,:3]@[0,0,-1] for d in list(ds["labels"].values())])
    imms = np.array([d for d in list(ds["inputs"].values())])
    smms = np.array([d for d in list(ds["static_inputs"].values())])

    # average over dataset
    mms = np.concatenate([imms, smms], axis=0)
    mmms = np.mean(mms, axis=(0,2))
    mmax = np.max(mmms)
    mmms /= mmax
    lm = upscale_repeat(np.expand_dims(mmms[0,:,:], 0), 10)
    rm = upscale_repeat(np.expand_dims(mmms[1,:,:], 0), 10)

    # just take a single frame from dataset
    # mmax = np.max(smms[0,0,0,:])
    # smms /= mmax
    # lm = upscale_repeat(np.expand_dims(smms[0,0,0,:], 0), 10)
    # rm = upscale_repeat(np.expand_dims(smms[0,1,0,:], 0), 10)

    lm = mm2img(lm)
    rm = mm2img(rm)

    div_h = rm.shape[1]
    div_w = int(0.05*rm.shape[2])
    div = np.reshape(np.repeat([255], 1*div_h*div_w*3), (1, div_h, div_w, 3))

    # combine matrices ...
    mm = np.squeeze(np.concatenate([lm,div,rm], axis=2).astype(np.uint8))

    fig = plt.figure(figsize=(9.71, 8.61))
    gs = gridspec.GridSpec(2,4)

    axp = AxesPlot(fig=fig, sps=gs[0,:2], angles=(-90,90))
    axp.plot_points(vecs, color="grey", label="rotation samples")
    axp.axtitle("Object Rotations (X-Y Plane)")

    axp2 = AxesPlot(fig=fig, sps=gs[0,2:], angles=(-90,0))
    axp2.plot_points(vecs, color="grey", label="rotation samples")
    axp2.axtitle("Object Rotations (X-Z Plane)")

    axm = fig.add_subplot(gs[1,1:3])
    axm.imshow(mm)
    axm.set_title(f"Mean Myrmex Activation [max={mmax:.4f}]")

    fig.suptitle(f"Dataset: {dsname}")
    # axp.store(f"{savepath}/analysis_{dsname}.png")
    axp.show()

if __name__ == "__main__":
    datasets = [
        DatasetName.cylinder_extreme,
        DatasetName.cuboid_extreme,
        # DatasetName.cylinder_large,
        # DatasetName.cuboid_large
    ]
    this_path = __file__.replace(__file__.split('/')[-1], '')
    plot_path = f"{this_path}/../plots/"
    for ds in datasets:
        ds_analysis_plot(ds, plot_path)