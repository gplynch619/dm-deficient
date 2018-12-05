import matplotlib.pyplot as plt
import numpy as np

with open('464.txt', 'r') as f:
    content = f.readlines()

arr = np.array([x.strip().split(' ') for x in content])
arr = arr.astype(int)

count = arr[:,1]
plt.hist(count, bins=10, range=[0,10])
plt.show()
