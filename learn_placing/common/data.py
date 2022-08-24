import os
import pickle

from datetime import datetime

def load_dataset(folder):
    ds = {}

    for fi in os.listdir(folder):
        if ".pkl" not in fi: continue
        fname = fi.replace(".pkl", "")
        if len(fname)<10:continue
        
        try:
            stamp = datetime.strptime(fname, "%Y-%m-%d.%H_%M_%S")
        except:
            stamp = datetime.strptime(fname, "%Y-%m-%d.%H:%M:%S")
        
        pkl = f"{folder}/{fi}"
        with open(pkl, "rb") as f:
            sample = pickle.load(f)
        ds.update({stamp: sample})
    return ds

def load_dataset_file(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
