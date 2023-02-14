import os
import torch
import numpy as np
import learn_placing.common.transformations as tf

from PIL import Image
from learn_placing.common.data import load_dataset
from learn_placing.common.myrmex_processing import preprocess_myrmex, mm2img, upscale_repeat

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from learn_placing.training.tactile_insertion_rl import TactilePlacingNet

from learn_placing.training.utils import DatasetName, get_dataset, load_train_params, rep2loss


def rpy_diff(out, lbl):
    # out = np.where(out<0, np.pi+out, out)
    # lbl = np.where(lbl<0, np.pi+lbl, lbl)
    return np.abs(out-lbl)


if __name__ == "__main__":
     # load neural net
    trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2023.02.13_18-45-21/CombinedAll/CombinedAll_Neps40_static_tactile_2023.02.13_18-45-21"
    # trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/ias_training_new_ds/Combined3D/Combined3D_Neps40_static_tactile_gripper_2022.09.13_10-42-03"
    trial_weights = f"{trial_path}/weights/best.pth"

    params = load_train_params(trial_path)
    train_l, test_l, _ = get_dataset(params.dsname, params, seed=params.dataset_seed)

    model  = TactilePlacingNet(**params.netp)
    crit   = rep2loss(params.loss_type)

    checkp = torch.load(trial_weights)
    model.load_state_dict(checkp)
    model.eval()

    nnerrs = np.array([])
    nnerrsy = []
    preds = np.array([])
    pcaerrs = np.array([])
    pcapreds = np.array([])
    outths = []

    with torch.no_grad():
        for data in test_l:
            inputs, grip, ft, lbls = data
            inputs = inputs[:,:,10,:,:] # get rid of sequence dim; PCA doesn't use it

            outs = model(inputs, grip, ft)
            loss_t = crit(outs, lbls)

            # test y-axis rotation only loss
            for pred, lbl in zip(outs, lbls):
                zlbl = lbl.numpy()@[0,0,-1]
                zpred = pred.numpy()@[0,0,-1]

                outth = np.arccos(np.dot(zpred, [0,0,-1]))
                lblth = np.arccos(np.dot(zlbl, [0,0,-1]))

                outths.append(outth)

                nnerrsy.append(np.abs(outth-lblth))

            nnerrs = np.concatenate([nnerrs, loss_t])
    nnerrsy = np.array(nnerrsy)
    outths = np.array(outths)

    print(nnerrs.shape, np.mean(nnerrs))
    print(np.mean(nnerrsy, axis=0))
    exit(0)

    """
    dataset:
    timestamp - sample (i.e. dict of time series)
        |-> tactile_left
            |-> [timestamps]
            |-> [myrmex samples]
    """
    
    # load sample 
    frame_no = 10
    sample_no = 188
    
    sample = ds[sample_no][1]
    mm, w2g, ft, lbl = extract_sample(sample)
    lblth = label_to_line_angle(lbl)
    # lblth = tf.euler_from_quaternion(lbl)[1]



    pred = model(*[torch.Tensor(np.array(x)) for x in [np.expand_dims(mm, 0), w2g, 0]])
    pred = np.squeeze(pred.detach().numpy())
    nnerr = single_pred_loss(pred, lbl, criterion)
    # nnth = label_to_line_angle(pred)
    nnth = tf.euler_from_matrix(pred)[1]

    """ NOTE this evaluation only respects the NN's accuracy of the rotation about the gripper frame's y-axis
    """
    nnerr = single_pred_loss(tf.Ry(nnth), tf.quaternion_from_euler(0,lblth,0), criterion)

    # perform PCA
    means, evl, evec, evth = predict_PCA(mm)
    evth = evth[0]
    pcaerr = single_pred_loss(tf.Ry(evth), tf.quaternion_from_euler(0,lblth,0), criterion)

    print()
    print(f"PCA err {pcaerr:.4f} | NN  err {nnerr:.4f}")
    print(f"PCA th  {evth:.4f} | NN  th {nnth:.4f} | LBL th {lblth:.4f}")
