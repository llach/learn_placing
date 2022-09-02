import os
from datetime import datetime

datefmt = "%Y.%m.%d_%H-%M-%S"
training_path = f"{os.environ['HOME']}/tud_datasets/trainings"

def now():  return datetime.now().strftime(datefmt).replace(':','-')