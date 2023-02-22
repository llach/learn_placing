import os
import pickle

from learn_placing.training.utils import DatasetName, InRot, ds2name


if __name__ == "__main__":
    
    dsnames = [DatasetName.upc_cub1, DatasetName.upc_cyl1]
    data_root = f"{os.environ['HOME']}/tud_datasets"
    for dd in dsnames: 
        dsname = ds2name[dd]

        print(f"processing dataset {dsname} ...")
        dataset_file = f"{data_root}/{dsname}.pkl"
        dataset_path = f"{data_root}/{dsname}/"

        ft = {}
        labels = {}
        inputs = {}
        
        for i, fi in enumerate(os.listdir(dataset_path)):
            if fi == "pics": continue
            
            with open(f"{dataset_path}{fi}", "rb") as f:
                d = pickle.load(f)

            ft |= {i: d["ft"]}
            inputs |= {i:d["mm"]}
            labels |= {i: {
                InRot.g2o: d["Qgo"],
                InRot.w2o: d["Qwo"],
                InRot.w2g: d["Qwg"]
            }}

        with open(dataset_file, "wb") as f:
            pickle.dump({
                "labels": labels, 
                "static_inputs": inputs,
                "static_ft": ft
            }, f)
    print("all done!")