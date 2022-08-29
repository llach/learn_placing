import torch 
import numpy as np

from learn_placing import training_path
from learn_placing.common.label_processing import rotate_v, normalize
from learn_placing.common.vecplot import AxesPlot

from utils import RotRepr, load_train_params, test_net, rep2loss, get_dataset
from tactile_insertion_rl import TactileInsertionRLNet

# trial_name = "Cuboid_Neps50_ortho6d_world2object_gripper-True_2022.08.27_20-46-32" 
# trial_name = "Cuboid_Neps50_ortho6d_world2object_gripper-True_2022.08.28_15-19-06" # gripper-only
# trial_name = "Cuboid_Neps50_ortho6d_world2object_cleanX_gripper-False_2022.08.29_14-59-28" # cleanX, static input
# trial_name = "Cuboid_Neps50_ortho6d_world2object_cleanX_gripper-False_2022.08.29_15-22-16" # cleanX, with tap
trial_name = "Cuboid_Neps50_ortho6d_world2object_cleanX_gripper-True_2022.08.29_16-10-11" #  cleanX, with tap, with gripper tf

trial_path = f"{training_path}/{trial_name}"
trial_weights = f"{trial_path}/weights/final.pth"

a = load_train_params(trial_path)

model = TactileInsertionRLNet(**a.netp)
checkp = torch.load(trial_weights)
model.load_state_dict(checkp)

criterion = rep2loss(a.out_repr)

(train_l, train_ind), (test_l, test_ind), _ = get_dataset(a.dsname, a, indices=[a.train_indices, a.test_indices, a.val_indices])

outputs, labels, loss, grips = test_net(model, criterion, test_l)
for i, out, lbl, lo, grip in zip(range(outputs.shape[0]), outputs, labels, np.squeeze(loss), grips):
    axp = AxesPlot()

    gv = rotate_v([0,0,-1], grip)
    if a.out_repr == RotRepr.quat:
        ov = rotate_v([0,0,-1], normalize(out))
        lv = rotate_v([0,0,-1], lbl)
    elif a.out_repr == RotRepr.ortho6d:
        ov = out@[0,0,-1]
        lv = lbl@[0,0,-1]

    axp.plot_v(gv, label=f"gripper rot", color="yellow")
    axp.plot_v(ov, label=f"out {lo:.5f}", color="black")
    axp.plot_v(lv, label="lbl", color="grey")
    axp.title(f"Test Sample {i+1}")
    axp.show()