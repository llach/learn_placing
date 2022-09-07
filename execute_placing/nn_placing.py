import os
import torch
import copy
import rospy
import pickle
import numpy as np

from tf import TransformListener, TransformBroadcaster
from threading import Lock
from placing_manager.srv import ExecutePlacing, ExecutePlacingResponse
from execute_placing.placing_planner import PlacingPlanner
from learn_placing import now
from learn_placing.training.utils import load_train_params, InData, rep2loss
from learn_placing.common.transformations import quaternion_from_matrix, quaternion_matrix
from learn_placing.training.tactile_insertion_rl import TactilePlacingNet
from learn_placing.processing.bag2pickle import msg2matrix, msg2ft
from learn_placing.processing.preprocess_dataset import myrmex_transform, ft_transform

class NNPlacing:
    grasping_frame = "gripper_left_grasping_frame"
    world_frame = "base_footprint"

    def __init__(self, trial_path, weights_name, store_samples = True) -> None:
        trial_weights = f"{trial_path}/weights/{weights_name}.pth"
        self.samples_dir = f"{os.environ['HOME']}/nn_samples/"
        self.net_name = trial_path.split("/")[-1]
        self.samples_net_dir = f"{self.samples_dir}/{self.net_name}"

        self.store_samples = store_samples
        if self.store_samples:
            os.makedirs(self.samples_dir)
            os.makedirs(self.samples_net_dir)

        self.params = load_train_params(trial_path)
        self.model = TactilePlacingNet(**self.params.netp)
        self.criterion = rep2loss(self.params.loss_type)

        self.olock = Lock()
        self.object_tf = None

        checkp = torch.load(trial_weights)
        self.model.load_state_dict(checkp)
        self.model.eval()

        self.planner = PlacingPlanner()
        self.placingsrv = rospy.Service("/nn_placing", ExecutePlacing, self.place)

        self.br = TransformBroadcaster()
        self.li = TransformListener()
        for _ in range(6):
            try:
                self.li.waitForTransform(self.grasping_frame, self.world_frame, rospy.Time(0), rospy.Duration(3))
                break
            except Exception as e:
                print(e)

    def pub_object_tf(self):
        with self.olock:
            if self.object_tf is None: return

            self.br.sendTransform(
                [0,0,0],
                quaternion_from_matrix(self.object_tf),
                rospy.Time.now(),
                "object_predicted",
                "base_footprint"
            )

    def place(self, req):
        print("placing object with NN ...")

        try:
            (_, Qwg) = self.li.lookupTransform(self.world_frame, self.grasping_frame, rospy.Time(0))
        except Exception as e:
            print(f"[ERROR] couldn't get gripper TF: {e}")
            return

        tleft = [msg2matrix(m) for m in req.tactile_left]
        tright = [msg2matrix(m) for m in req.tactile_right]
        ft = [msg2ft(m) for m in req.ft]

        tinp, tinp_static = myrmex_transform(tleft, tright, self.params.dsname)
        ftinp, ftinp_static = ft_transform(ft, self.params.dsname)

        """
        x.shape
        torch.Size([8, 2, 50, 16, 16])
        -> [batch, sensors, sequence, H, W]

        gr.shape
        torch.Size([8, 4])
        -> [batch, Q]

        ft.shape
        torch.Size([8, 15, 6])
        -> [batch, sequence, FT]
        """
        if self.params.input_data == InData.static:
            print(self.params.input_data)
            xs = [[tinp_static], [Qwg], [ftinp_static]]
        elif self.params.input_data == InData.with_tap:
            print(self.params.input_data)
            xs = [[tinp], [Qwg], [ftinp]]
        prediction = self.model(*[torch.Tensor(np.array(x)) for x in xs])
        prediction = np.squeeze(prediction.detach().numpy())

        try:
            (_, Qwo) = self.li.lookupTransform(self.world_frame, "object", rospy.Time(0))
            loss = self.criterion(torch.Tensor([prediction]), torch.Tensor([quaternion_matrix(Qwo)[:3,:3]]))
            print(f"loss: {loss}")
        except Exception as e:
            loss = None

        if self.store_samples:
            with open(f"{self.samples_net_dir}/{now()}.pkl", "wb") as f:
                pickle.dump({
                    "xs": xs,
                    "y": prediction,
                    "loss": loss 
                })

        Tpred = np.eye(4)
        Tpred[:3,:3] = prediction
        print(Tpred)

        with self.olock:
            self.object_tf = copy.deepcopy(Tpred)

        # print("aligning object ...")
        # done = False
        # while not done:
        #     inp = input("next? a=align; p=place\n")
        #     inp = inp.lower()
        #     if inp == "a":
        #         self.planner.align(Tpred)
        #     elif inp == "p":
        #         self.planner.place()
        #         break
        #     else:
        #         done = True
        # print("all done, bye")

        # used to debug the input
        # self.plot_input(tinp, tinp_static, ftinp, ftinp_static)

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

        msmin, msmax = np.min(smyr), np.max(smyr)
        fsmin, fsmax = np.min(sft), np.max(sft)

        fmin, fmax = np.min([fmin, fsmin]), np.max([fmax, fsmax])
        mmin, mmax = np.min([mmin, msmin]), np.max([mmax, msmax])
        
        axs[0,0].plot(range(len(sft)),sft)
        axs[0,1].plot(range(len(ft)),ft)
        for ax in axs[0,:]: ax.set_ylim(0.9*fmin, 1.1*fmax)

        axs[1,0].plot(range(len(smyr)),smyr)
        axs[1,1].plot(range(len(myr)),myr)
        for ax in axs[1,:]: ax.set_ylim(0.9*mmin, 1.1*mmax)

        axs[0,0].set_title("FT static")
        axs[0,1].set_title("FT dynamic")

        axs[1,0].set_title("Tactile static")
        axs[1,1].set_title("Tactile dynamic")

        fig.suptitle(f"Input Data in NN Node")
        fig.tight_layout()
        plt.show()
    

if __name__ == "__main__":
    rospy.init_node("nn_placing")

    netname = "/home/llach/tud_datasets/batch_trainings/2022.09.07_11-14-24/GripperVar2/GripperVar2_Neps20_static_tactile_2022.09.07_12-48-15"
    # netname = "/home/llach/tud_datasets/batch_trainings/2022.09.03_11-13-08/ObjectVar/ObjectVar_Neps20_tactile_2022.09.03_11-13-08"
    weights = "final"

    nnp = NNPlacing(netname, weights)
    r = rospy.Rate(20)

    while not rospy.is_shutdown():
        nnp.pub_object_tf()
        r.sleep()
    print("bye!")