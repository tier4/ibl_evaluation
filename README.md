# ROS Toolbox

This repository provides the toolbox for tier4 rosbag decompression, IBL preprocessing and IBL evaluation.

## Dependency

- See requirements.txt.

## Usage

Run `decompress_rosbag.py` to extract images from Tier4-format rosbags in a batch manner.

```python3
python decompress_rosbag.py
```

Configure the path information in config files and run scipts to preprocess the workspace.

```python3
# Tier4 dataset
python preprocess_tier4.py

# CMU dataset
python preprocess_cmu.py

# RobotCar dataset
python preprocess_robotcar.py
```

Run `evaluate_ibl.py` to evaluate IBL results.

```python3
python evaluate_ibl.py
```

## Important Settings

### Tier4 dataset

- input_dir
- origin
- T_sensor_cam
- cam_conf

### CMU dataset

- input_dir
- slice (FIXME: the information of slice is usually included in the `input_dir`)

### RobotCar dataset

- input_dir
