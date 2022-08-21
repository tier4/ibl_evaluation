import os
from pathlib import Path
import argparse
import numpy as np
import cv2
from easydict import EasyDict as edict
from rclpy.serialization import deserialize_message
from rosbags.rosbag2 import Reader
from rosidl_runtime_py.utilities import get_message

from utils import load_yaml

file_dir = os.path.dirname(os.path.abspath(__file__))


def get_filter(topic, reader):
    filter_ = []
    for connection, _, _ in reader.messages():
        if connection.topic == topic:
            filter_.append(connection)
            return filter_

    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        type=str,
                        help='config file for ros2bag decompression',
                        default=os.path.join(file_dir, 'configs/decompress_ros2bag_config.yaml'))
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    conf_dict = edict(load_yaml(args.config))
    output_dir = Path(conf_dict.output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)

    reader = Reader(conf_dict.input_dir)
    reader.open()
    filter_ = get_filter(conf_dict.topic, reader)
    for connection, timestamp, rawdata in reader.messages(filter_):
        imu_data = deserialize_message(rawdata, get_message(connection.msgtype))
        np_arr = np.frombuffer(imu_data.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if conf_dict.is_BGR:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(str(output_dir / '{}.jpg'.format(timestamp)), image)

    reader.close()


if __name__ == '__main__':
    main()
