import os
import getpass

address0 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/'
address1 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/interaction/'
address2 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/task-level/'
address3 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/interaction/setting/'
address4 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/interaction/results/'
address5 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/'
address6 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/noCom/'
address7 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/noCom/no_collision/'
address8 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/data_results/noCom/collision/'

n_iter = 1000

for i in range(n_iter):
    old_name = 'th0.8_result' + '_' + str(i) + '.txt'
    new_name = 'th0.8_result' + '_' + str(i+50) + '.txt'
    if os.path.exists(address5 + old_name):
        os.rename(address5 + old_name, address5 + new_name)