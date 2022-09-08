import os
import torch 
import numpy as np

from learn_placing import training_path
from torch.utils.data import TensorDataset, DataLoader
from learn_placing.common import load_dataset
from learn_placing.common.data import load_dataset_file
from learn_placing.common.transformations import quaternion_from_matrix, quaternion_inverse
from learn_placing.training.tactile_insertion_rl import TactilePlacingNet
from learn_placing.training.utils import DatasetName, get_dataset, load_train_params, rep2loss, test_net


def l2T(ll): return [torch.Tensor(np.array(l)) for l in ll]

if __name__ == "__main__":
    dsname = DatasetName.gripper_var2 
    # modelname =  "GripperVar2_Neps20_static_gripper_2022.09.07_12-51-46"
    modelname =  "GripperVar2_Neps20_static_ft_2022.09.07_12-51-54"

    data_root = f"{os.environ['HOME']}/tud_datasets/nn_samples/"
    dataset_path = f"{data_root}/GripperVar2_Neps20_static_gripper_2022.09.07_12-51-46"

    # dsfilename = "gripper_trial"
    # dataset_file_path = f"{os.environ['HOME']}/tud_datasets/{dsfilename}.pkl"

    # sample timestamp -> sample
    dsnn = load_dataset(dataset_path)
    dsnn = dict([(sk, dsnn[sk]) for sk in sorted(dsnn)])

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
    labels = np.array(labels)

    tds = TensorDataset(*[torch.Tensor(np.array(u)) for u in [mms, gps, fts, labels]])
    nn_in = DataLoader(tds, shuffle=False, batch_size=8)

    trial_path = f"{training_path}/../batch_trainings/2022.09.07_11-14-24/GripperVar2/{modelname}"
    trial_weights = f"{trial_path}/weights/final.pth"

    a = load_train_params(trial_path)

    model = TactilePlacingNet(**a.netp)
    checkp = torch.load(trial_weights)
    model.load_state_dict(checkp)

    criterion = rep2loss(a.loss_type)

    _, test_ds, _ = get_dataset("gripper_trial", a, seed=a.dataset_seed, train_ratio=0.0, batch_size=16, random=False)
    dsout, dslbl, dsloss = [], [], []
    batch = list(test_ds)[0]
    for sample in zip(*batch):
        dx, dgr, dft, dy = sample
        with torch.no_grad():
            # print(ft.numpy())
            dft = dft.unsqueeze(0)
            print(dft[0,0,0])
            # print(dft.shape)
            ou = model(None, None, dft)
            # if dft[0,0,0].numpy()==-12.5000:
            #     print(dy.numpy())
            dsout.append(ou.numpy())
            dslbl.append(dy.numpy())
            T = np.eye(4)
            T[:3,:3] = dsout[-1]
            dsloss.append(np.squeeze(criterion(ou, dy.unsqueeze(0)).numpy()))
    print(np.array(dsloss))

    out, lbl, nn_loss = [], [], []
    for i, x, gr, ft, y in zip(range(len(mms)), mms, gps, fts, labels):
        # print(ft)
        with torch.no_grad():
            ft = torch.Tensor(np.array([ft]))
            print(ft[0,0,0])
            # print(ft.shape)
            ou = np.squeeze(model(None,None,ft)).numpy()
            out.append(ou)
            # if ft[0,0,0].numpy()==-12.5000:
            #     print(y)
            lbl.append(y)
            nn_loss.append(criterion(*l2T([[ou],batch[-1][i].unsqueeze(0)])).numpy())

    # for o,l in zip(out, lbl):
    #     print(f"out\n{o}")
    #     print(f"lbl\n{l}")

    print(np.array(nn_loss))



    pass