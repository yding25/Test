import numpy as np

p_jam = 0.5
num_1 = 0
num_0 = 0
for i in range(100):
    jam = np.random.choice(np.arange(0, 2), p=[p_jam, 1 - p_jam])
    if jam == 0:
        num_0 += 1
    else:
        num_1 += 1

print(num_0)
print(num_1)