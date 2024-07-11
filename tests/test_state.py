import optimizer
import numpy as np
from matplotlib import pyplot as plt

crowding_distance = True
position = False

# Testing crowding distance
if crowding_distance:
    optimizer.FileManager.working_dir = "tmp/zdt1/"
    distances_42 = optimizer.FileManager.load_csv("crowding_distances_42.csv").tolist()
    distances_42 =  np.transpose([row[1:-1] for row in distances_42])
    mean_42 = np.mean(distances_42, axis = 0)
    distances_43 = optimizer.FileManager.load_csv("crowding_distances_43.csv").tolist()
    distances_43 =  np.transpose([row[1:-1] for row in distances_43])
    mean_43 = np.mean(distances_43, axis = 0)
    plt.figure()
    plt.plot(mean_42)
    plt.plot(mean_43)
    plt.show()

# # Position
# if position:



