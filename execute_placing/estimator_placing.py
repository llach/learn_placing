import os 
import rospy

from tf import TransformListener
from run_estimator import RunEstimators
from placing_planner import PlacingPlanner
from placing_manager.srv import EstimatorPlacing, EstimatorPlacingResponse
from learn_placing.common import tft

class EstimatorPlacingService:
    grasping_frame = "gripper_grasping_frame"
    world_frame = "base_footprint"

    def __init__(self, trial_path, noise_thresh) -> None:
        self.rune = RunEstimators(trial_path, noise_thresh=noise_thresh, publish_image=True)

        self.planner = PlacingPlanner()
        self.placingsrv = rospy.Service("/placing", EstimatorPlacing, self.place)

        self.li = TransformListener()
        self.li.waitForTransform(self.grasping_frame, self.world_frame, rospy.Time(0), rospy.Duration(3))

    def place(self, req):
        # prepare data
        model = req.model
        results = self.rune.estimate()
        suc_models = list(results.keys())

        # sanity check: do we have data from the requested model?
        if model not in suc_models:
            print(f"{model} was not in {suc_models}. ABORTING")
            return EstimatorPlacingResponse()

        # pointers to transform and error
        Rgo = results[model][0][:3,:3]
        err = results[model][1]
        
        # planner expects Rwo, so we calculate Rwg x Rgo
        (_, Qwg) = self.li.lookupTransform(self.world_frame, self.grasping_frame, rospy.Time(0))
        Rwo = tft.quaternion_matrix(Qwg)[:3,:3].dot(Rgo)

        print(f"placing model {model} with error {err}")
        while True:
            inp = input("next? a=align; p=place\n")
            inp = inp.lower()
            if inp == "a":
                self.planner.align(Rwo)
            elif inp == "p":
                self.planner.place()
                break
            else: break

        print("all done, bye")
        return EstimatorPlacingResponse(err=err)
    

if __name__ == "__main__":
    noise_thresh = 0.05
    trial_path = f"{os.environ['HOME']}/tud_datasets/batch_trainings/2023.02.24_10-41-09/UPC_v1/UPC_v1_Neps60_static_tactile_2023.02.24_10-41-09"

    rospy.init_node("estimator_placing")

    ep = EstimatorPlacingService(trial_path, noise_thresh=noise_thresh)
    while not rospy.is_shutdown(): rospy.spin()