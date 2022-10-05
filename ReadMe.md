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

## Train Vision-Based Safety Estimator
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

Start training (an pretrained model titled pretrained_safety_model_town05_ANN.pt: https://www.dropbox.com/scl/fo/3cxbsuzxiak6qnbd7atiz/h?dl=0&rlkey=3o04anx3dkdg2j9eo8ksxa972)
```
Step 5 (option 1): train_ANN.py
Step 5 (option 2): train_SVM.py
```


## Run vision-based GLAD
Generate service request
```
Step 1: get_request.py
```

Get all tasks plans locally 
```
Step 2: get_taskplan_library.py
```
Note that just run Step 1 and Step 2 one time before each trial

Evaluate safety estimator's performance
```
Step 3: FP_TP_FN_TN.py
```

```
Step 4: main_abstract.py
```
