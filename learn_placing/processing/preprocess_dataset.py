import os
import pickle

from learn_placing import dataset_path
from learn_placing.training.utils import DatasetName, InRot, ds2name


if __name__ == "__main__":
    
    dsnames = [DatasetName.cub23, DatasetName.cyl23]
    for dd in dsnames: 
        dsname = ds2name[dd]

        print(f"processing dataset {dsname} ...")
        dataset_file = f"{dataset_path}/{dsname}.pkl"
        dataset_dir  = f"{dataset_path}/{dsname}/"

        ft = {}
        labels = {}
        inputs = {}
        
        for i, fi in enumerate(os.listdir(dataset_dir)):
            if fi == "pics": continue
            
            with open(f"{dataset_dir}{fi}", "rb") as f:
                d = pickle.load(f)

            ft.update({i: d["ft"]})
            inputs.update({i:d["mm"]})
            labels.update({i: {
                InRot.g2o: d["Qgo"],
                InRot.w2o: d["Qwo"],
                InRot.w2g: d["Qwg"]
            }})

        with open(dataset_file, "wb") as f:
            pickle.dump({
                "labels": labels, 
                "static_inputs": inputs,
                "static_ft": ft
            }, f)
    print("all done!")