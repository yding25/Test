import os
import getpass

# -----------------------------------------
# definite paths
# -----------------------------------------
address0 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/'
address1 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/interaction/'
address2 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/task-level/'
address3 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/interaction/setting/'
address4 = '/home/' + getpass.getuser() + '/CARLA_0.9.10.1/PythonAPI/Test/interaction/results/'

command1 = 'python ' + address2 + 'get_cost_risk_of_lane.py'
os.system(command1)

command2 = 'python ' + address0 + 'empty_milestones.py'
os.system(command2)

command3 = 'python ' + address0 + 'get_optimal_task_plan.py'
os.system(command3)

command4 = 'python ' + address2 + 'ground_task_plan.py'
os.system(command4)

command5 = 'python ' + address0 + 'main_abstract_our.py'
os.system(command5)
