import numpy as np
from matplotlib import pyplot as plt

rew = np.load("./models/model_normalized_hv_times_120_long/model_normalized_hv_120_mean_reward.npy")
plt.figure()
plt.plot(rew)
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.savefig("try_reward.png")

print(rew)