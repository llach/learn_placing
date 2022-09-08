import torch 
import numpy as np

from learn_placing import training_path
from learn_placing.common.label_processing import rotate_v, normalize
from learn_placing.common.vecplot import AxesPlot

from utils import RotRepr, load_train_params, test_net, rep2loss, get_dataset
from tactile_insertion_rl import TactilePlacingNet

# trial_name = "Cuboid_Neps50_ortho6d_world2object_gripper-True_2022.08.27_20-46-32" 
# trial_name = "Cuboid_Neps50_ortho6d_world2object_gripper-True_2022.08.28_15-19-06" # gripper-only
# trial_name = "Cuboid_Neps50_ortho6d_world2object_cleanX_gripper-False_2022.08.29_14-59-28" # cleanX, static input
# trial_name = "Cuboid_Neps50_ortho6d_world2object_cleanX_gripper-False_2022.08.29_15-22-16" # cleanX, with tap
# trial_name = "Cuboid_Neps50_ortho6d_world2object_cleanX_gripper-True_2022.08.29_16-10-11" #  cleanX, with tap, with gripper tf
# trial_name = "Cuboid_Neps50_sincos_local_dotproduct_gripper-False_2022.08.29_19-38-50" # world->object in gripper, WITH tap
# trial_name = "Cuboid_Neps50_sincos_local_dotproduct_gripper-False_2022.08.29_19-46-11" # world->object in gripper, STATIC
# trial_name = "ObjectVar_Neps10_sincos_local_dotproduct_gripper-True_2022.08.31_19-31-47"

### new datasets

# trial_name = "GripperVar_Neps10_ortho6d_world2object_gripper-False_2022.09.01_16-55-48"
# trial_name = "GripperVar_Neps10_ortho6d_world2object_gripper-False_2022.09.01_16-58-42"

trial_name = "../batch_trainings/2022.09.08_16-42-27/OptiGripperTest/OptiGripperTest_Neps20_static_tactile_2022.09.08_16-42-28"

trial_path = f"{training_path}/{trial_name}"
trial_weights = f"{trial_path}/weights/final.pth"

a = load_train_params(trial_path)

model = TactilePlacingNet(**a.netp)
checkp = torch.load(trial_weights)
model.load_state_dict(checkp)

criterion = rep2loss(a.loss_type)

train_l, test_l, _ = get_dataset("test", a, seed=a.dataset_seed, batch_size=3)

outputs, labels, loss, grips = test_net(model, criterion, test_l)
if a.out_repr == RotRepr.sincos:
    import matplotlib.pyplot as plt

    for i, out, lbl, lo, grip in zip(range(outputs.shape[0]), outputs, labels, np.squeeze(loss), grips):
        plt.scatter(*out, label=f"prediction | {lo:.5f}", color="black") # TODO use sum or mean here?
        plt.scatter(*lbl, label="label", color="grey")

        plt.xlim([-1.05, 1.05])
        plt.ylim([-1.05, 1.05])

        plt.title("Predicted Object to World Angle in Gripper Frame")
        plt.xlabel("cos(theta)")
        plt.ylabel("sin(theta)")

        plt.legend(loc="lower left")
        plt.show()
    exit(0)

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
    axp.plot_v(ov, label=f"prediction | {lo:.5f}", color="black")
    axp.plot_v(lv, label="label", color="grey")
    axp.title(f"Test Sample {i+1}")
    axp.show()