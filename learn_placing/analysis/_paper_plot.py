import pickle
import numpy as np
import matplotlib.pyplot as plt

from learn_placing import training_path

base_path = f"{training_path}/../batch_trainings/ias_training_new_ds/Combined3D/"
ft_path = "Combined3D_Neps40_static_ft_2022.09.13_18-52-44"
tac_path = "Combined3D_Neps40_static_tactile_2022.09.13_10-41-43"
tac_ft_path = "Combined3D_Neps40_static_tactile_ft_2022.09.13_10-42-33"

# with open(f"{base_path}/{ft_path}/losses.pkl", "rb") as f:
#     ft_loss = np.array(pickle.load(f)["test"])

with open(f"{base_path}/{tac_path}/losses.pkl", "rb") as f:
    tac_loss = np.array(pickle.load(f)["test"])

with open(f"{base_path}/{tac_ft_path}/losses.pkl", "rb") as f:
    tac_ft_loss = np.array(pickle.load(f)["test"])


fig, ax = plt.subplots(figsize=(7.5, 5))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

xs = np.arange(len(tac_loss))
plt.plot(np.mean(tac_loss, axis=1), label="Tactile-only")
plt.plot(np.mean(tac_ft_loss, axis=1), label="Tactile & F/T")

plt.ylabel("loss (rad)")
plt.ylim((0,np.pi))

plt.xlabel("batch")

plt.legend()
plt.tight_layout()
plt.show()
pass