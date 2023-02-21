from run_estimator import RunEstimators

class EstimatorPlacing:

    def __init__(self, trial_path) -> None:
        self.rune = RunEstimators(trial_path, noise_thresh=0.1, publihs_image=True)

    def place(self, model):
        results = self.rune.estimate()
        suc_models = list(results.keys())

        if model not in suc_models:
            print(f"{model} was not in {suc_models}. ABORTING")
            return False

        Tpred = results[model][0]
        err = results[model][1]

        print(f"placing model {model} with error {err}")
            
        while True:
            inp = input("next? a=align; p=place\n")
            inp = inp.lower()
            if inp == "a":
                self.planner.align(Tpred)
            elif inp == "p":
                self.planner.place()
                break
            else: break

        print("all done, bye")

if __name__ == "__main__":
    ep = EstimatorPlacing(9)