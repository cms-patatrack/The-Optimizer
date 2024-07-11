from optimizer.metrics import hyper_volume
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import time
from pymoo.indicators.hv import HV

hv_times = []
ref_point = [1,1]
# for i in range(20,1000,20):
#     print(i)
#     pareto_x = (np.linspace(0.1, 1, i))
#     pareto_y = 1/pareto_x
#     pareto = np.column_stack((pareto_x, pareto_y))

#     start = time.time()
#     hv = hyper_volume(pareto, ref_point)
#     end = time.time()
#     diff = end - start
#     hv_times.append(diff)
pareto = np.load("/home/marco/work/The-Optimizer/wrong_pareto.npy")
print("len ", len(pareto))

plt.figure()    
plt.scatter(pareto[:,0], pareto[:,1])
plt.show()
start = time.time()
ind = HV(ref_point=ref_point)
hv = ind(pareto)
end = time.time()
diff = end - start
print(diff)