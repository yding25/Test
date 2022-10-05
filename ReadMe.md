# Test (temporary name)

## Environment
ubuntu (version 18.04 has been tested) and CARLA (version 0.9.10 has been tested)

## Install Test

```
Step 1: Download and install CARLA
```
Detailed instruction can be found in https://carla.readthedocs.io/en/0.9.10/start_quickstart/

```
Step 2: Download 'Test' and place it in CARLA's PythonAPI folder
```

```
Step 3: Install necessary packages (pytorch, networkx, cv2, etc)
```
pytorch == 1.12.0, networkx = 2.6.3, cv2 = 4.6.0

## Part 1: Train Vision-Based Safety Estimator
Enter CARLA simulator
```
Step 1: run ./CarlaUE4.sh
```

Collect image data (an existing dataset: https://www.dropbox.com/scl/fo/3cxbsuzxiak6qnbd7atiz/h?dl=0&rlkey=3o04anx3dkdg2j9eo8ksxa972)
```
Step 2: collect_data.py
```

Split data into train and test
```
Step 3: split_data.py
```

Generate X (input) and Y (output)
```
Step 4: process_data.py
```

Start training
```
Step 5 (option 1): train_ANN.py
```
```
Step 5 (option 2): train_SVM.py
```

## Part 2: running vision-based TMPUD (abstract simualation)
```
Note: just need one time before each trial
Step 1: get_request.py
Step 2: get_taskplan_library.py
```

```
Step 3: FP_TP_FN_TN.py # evaluate the safety model performance
```

```
Step 4: main_abstract.py
```

## Part 3: plot
```
utility_bar_plot.py
```

## Part 4: running vision-based TMPUD (full simualation)

## Part 5: tools (including replay log, plot figure, etc)
```
Tool 1: other_cars.py # get all information of other cars
```

```
Tool 2: change_name_files_in_folder.py # change name of N files in a folder
```

```
Tool 3: change_name_files_in_txt.py # change name of N files in a txt
```

```
Tool 4: repeatN_code.py # repeatedly run a code file
```

```
Tool 5: repeatN_multiple_codes.py # repeatedly run a set of codes file
```

```
Tool 6: replay_log.py # replay log file
```

```
Tool 7: safety_bar_plot.py
# compare performance of five methods: ANN, SVM, AdaBoost, KNN, and LR
```

```
Tool 8: safety_curve_plot.py
# compare performance of two methods: ANN, SVM
```
