import numpy as np
import os
import matplotlib.pyplot as plt


path = r"../output"
fn = r"optimization_diagnostics_20251220_155452.csv"

with open(os.path.join(path, fn)) as infile:
    data_raw = np.loadtxt(infile, skiprows=1, delimiter=",", usecols=[0, 5, 6, 7])

data = data_raw[np.where(data_raw[:, 3] >= 0)]
data = data[np.where(data[:, 3] < 1e5)]

it = data[:, 0]
flat = data[:, 1]
reg = data[:, 2]
obj = data[:, 3]

fig, ax = plt.subplots(1, 2)

plt.sca(ax[0])

plt.plot(it, flat, label="Flatness")
plt.legend(loc=0)
plt.twinx()
plt.plot(it, reg, label="Regularization", color='red')
# plt.ylim(0.035, 0.04)
plt.legend(loc=1)

plt.sca(ax[1])
plt.plot(it, obj, label="Flatness")
plt.xlabel("Iterations")
plt.ylabel("Objective")

plt.show()
