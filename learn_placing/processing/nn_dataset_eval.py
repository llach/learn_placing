import os
import torch 
import numpy as np

from learn_placing import training_path
from learn_placing.common import load_dataset
from learn_placing.common.data import load_dataset_file
from torch.utils.data import TensorDataset, DataLoader
from learn_placing.training.tactile_insertion_rl import TactilePlacingNet
from learn_placing.training.utils import DatasetName, get_dataset, load_train_params, rep2loss, test_net


if __name__ == "__main__":
    dsname = DatasetName.gripper_var2 
    modelname =  "GripperVar2_Neps20_static_gripper_2022.09.07_12-51-46"

    data_root = f"{os.environ['HOME']}/nn_samples/"
    dataset_path = f"{data_root}/{modelname}"

    dsfilename = "gripper_trial"
    dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{dsfilename}.pkl"
    ds = load_dataset_file(dataset_file_path)

    # sample timestamp -> sample
    dsnn = load_dataset(dataset_path)

    mms = []
    gps = []
    fts = []
    labels = []
    for _, v in dsnn.items():
        mms.append(v["xs"][0][0])
        gps.append(v["xs"][1][0])
        fts.append(v["xs"][2][0])
        labels.append(v["y"])
    mms = np.array(mms)
    gps = np.array(gps)
    fts = np.array(fts)

    tds = TensorDataset(*[torch.Tensor(np.array(u)) for u in [mms, gps, fts, labels]])
    nn_in = DataLoader(tds, shuffle=False, batch_size=8)

    trial_name = "/GripperVar2/GripperVar2_Neps20_static_gripper_2022.09.07_12-51-46"

    trial_path = f"{training_path}/../batch_trainings/2022.09.07_11-14-24/GripperVar2/{modelname}"
    trial_weights = f"{trial_path}/weights/final.pth"

    a = load_train_params(trial_path)

    model = TactilePlacingNet(**a.netp)
    checkp = torch.load(trial_weights)
    model.load_state_dict(checkp)

    criterion = rep2loss(a.loss_type)

    _, test_ds, _ = get_dataset("gripper_trial", a, seed=a.dataset_seed, train_ratio=0.0)
    _, _, ds_loss, _ = test_net(model, criterion, test_ds)

    print(ds_loss)

    out, lbl, nn_loss = [], [], []
    


    for o,l in zip(out, lbl):
        print(f"out\n{o}")
        print(f"lbl\n{l}")

    print(nn_loss)



    pass