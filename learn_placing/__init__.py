import os
from datetime import datetime

datefmt = "%Y.%m.%d_%H-%M-%S"
training_path = f"{os.environ['HOME']}/tactile_placing/trainings"
dataset_path = f"{os.environ['HOME']}/tactile_placing/"

def now():  return datetime.now().strftime(datefmt).replace(':','-')