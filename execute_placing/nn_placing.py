import torch
import rospy
import numpy as np

from placing_manager.srv import ExecutePlacing, ExecutePlacingResponse

from learn_placing.training.utils import load_train_params
from learn_placing.training.tactile_insertion_rl import TactilePlacingNet
from learn_placing.processing.bag2pickle import msg2matrix, msg2ft
from learn_placing.processing.preprocess_dataset import myrmex_transform, ft_transform

class NNPlacing:

    def __init__(self, trial_path, weights_name) -> None:
        trial_weights = f"{trial_path}/weights/{weights_name}.pth"

        self.params = load_train_params(trial_path)
        self.model = TactilePlacingNet(**self.params.netp)

        checkp = torch.load(trial_weights)
        self.model.load_state_dict(checkp)
        self.model.eval()

        self.placingsrv = rospy.Service("/nn_placing", ExecutePlacing, self.place)

    def place(self, req):
        print("placing object with NN ...")

        tleft = [msg2matrix(m) for m in req.tactile_left]
        tright = [msg2matrix(m) for m in req.tactile_right]
        ft = [msg2ft(m) for m in req.ft]

        tinp, tinp_static = myrmex_transform(tleft, tright, self.params.dsname)
        ftinp, ftinp_static = ft_transform(ft, self.params.dsname)

        self.plot_input(tinp, tinp_static, ftinp, ftinp_static)

        return ExecutePlacingResponse()

    def plot_input(self, tinp, tinp_static, ftinp, ftinp_static):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(9.71, 8.61))

        ft = np.mean(ftinp, axis=-1)
        sft = np.mean(ftinp_static, axis=-1)

        myr = np.swapaxes(tinp, 0, 1)
        smyr = np.swapaxes(tinp_static, 0, 1)

        myr = np.mean(myr, axis=(1,2,3))
        smyr = np.mean(smyr, axis=(1,2,3))

        mmin, mmax = np.min(myr), np.max(myr)
        fmin, fmax = np.min(ft), np.max(ft)
        
        axs[0,0].plot(range(len(sft)),sft)
        axs[0,1].plot(range(len(ft)),ft)
        for ax in axs[0,:]: ax.set_ylim(1.1*fmin, 1.1*fmax)

        axs[1,0].plot(range(len(smyr)),smyr)
        axs[1,1].plot(range(len(myr)),myr)
        for ax in axs[1,:]: ax.set_ylim(1.1*mmin, 1.1*mmax)

        axs[0,0].set_title("FT static")
        axs[0,1].set_title("FT dynamic")

        axs[1,0].set_title("Tactile static")
        axs[1,1].set_title("Tactile dynamic")

        fig.suptitle(f"Input Data in NN Node")
        fig.tight_layout()
        plt.show()
    

if __name__ == "__main__":
    rospy.init_node("nn_placing")

    netname = "/home/llach/tud_datasets/batch_trainings/2022.09.02_11-38-14/ObjectVar/ObjectVar_Neps20_tactile_gripper_ft_2022.09.02_11-55-10"
    weights = "final"

    nnp = NNPlacing(netname, weights)
    rospy.spin()