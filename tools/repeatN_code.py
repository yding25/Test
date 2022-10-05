import subprocess
import sys
import time

init_time = time.time()
script_name = '/home/yan/CARLA_0.9.10.1/PythonAPI/Test/safety_estimator/collect_data.py'
n_iter = 100
for i in range(n_iter):
    print('epoch:{}'.format(i))
    subprocess.call(['python', script_name], stdout=sys.stdout, stderr=subprocess.STDOUT)

print('computing time cost (s):{}'.format(time.time() - init_time))