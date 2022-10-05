import numpy as np
import matplotlib as plt

file = open('/home/yan/CARLA_0.9.10.1/PythonAPI/Test/loss.txt', 'r')
data = []
for line in file.readlines():
   data.append(float(line))
time = range(0, len(data))
print(data)
plt.plot(time, data)
plt.xlabel('Time (hr)')
plt.ylabel('Position (km)')
plt.show()