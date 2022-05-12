# ROS Toolbox

This repository provides the toolbox for tier4 rosbag decompression, IBL preprocessing and IBL evaluation.

## Dependency

- See requirements.txt.

## Usage

Run `decompress_rosbag.py` to extract images from Tier4-format rosbags in a batch manner.

```python3
python decompress_rosbag.py
```

Run `preprocess.py` to preprocess the workspace.

```python3
python preprocess_tier4.py
```

Run `evaluate_ibl.py` to evaluate IBL results.

```python3
python evaluate_ibl.py
```

## Important Settings

- origin
- T_sensor_cam
- cam_conf
