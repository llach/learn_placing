import torch 
import numpy as np
import matplotlib.pyplot as plt

from learn_placing import training_path
from learn_placing.common.label_processing import rotate_v, normalize
from learn_placing.common.vecplot import AxesPlot

from utils import load_train_params, test_net, rep2loss, get_dataset
from tactile_insertion_rl import TactilePlacingNet

# trial_name = "../batch_trainings/2022.09.07_11-14-24/ObjectVar2/ObjectVar2_Neps20_static_tactile_2022.09.07_12-31-22"
trial_name = "../batch_trainings/2022.09.08_16-42-27/OptiGripperTest/OptiGripperTest_Neps20_static_tactile_2022.09.08_16-42-28"

trial_path = f"{training_path}/{trial_name}"
trial_weights = f"{trial_path}/weights/final.pth"

a = load_train_params(trial_path)

model = TactilePlacingNet(**a.netp)
checkp = torch.load(trial_weights)
model.load_state_dict(checkp)

criterion = rep2loss(a.loss_type)

test_l, _, _ = get_dataset("test", a, seed=a.dataset_seed, train_ratio=1.0, random=False)

outputs, labels, loss, grips = test_net(model, criterion, test_l)
cm = plt.get_cmap("copper")
axp = AxesPlot()
for i, out, lbl, lo, grip in zip(range(outputs.shape[0]), outputs, labels, np.squeeze(loss), grips):
    lo_norm = lo/np.pi
    
    gv = rotate_v([0,0,-1], grip)
    ov = out@[0,0,-1]
    lv = lbl@[0,0,-1]

    axp.plot_v(lv, color=cm(lo_norm))
print(loss)

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

cmappable = ScalarMappable(norm=Normalize(0,1), cmap="copper")
axp.fig.colorbar(cmappable, ax=axp.ax)

axp.title(f"Labels colorized by loss magnitude")
axp.show()