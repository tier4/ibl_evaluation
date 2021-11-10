import os
from pathlib import Path
import argparse
import glob

import numpy as np
import cv2
from easydict import EasyDict as edict
import rosbag

from utils import load_yaml

file_dir = os.path.dirname(os.path.abspath(__file__))


def decompress_image(ros_image_compressed, is_BGR=False):
    try:
        np_arr = np.frombuffer(ros_image_compressed.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if is_BGR:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(e)
    
    return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        type=str,
                        help='config file for rosbag decompression',
                        default=os.path.join(file_dir, 'decompress_rosbag_config.yaml'))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    conf_dict = edict(load_yaml(args.config))
    output_dir = Path(conf_dict.output_dir)

    rosbag_name_list = glob.glob(conf_dict.input_dir + '/{}*.bag'.format(conf_dict.prefix))
    print('Find {} rosbags.'.format(len(rosbag_name_list)))

    for rosbag_name in sorted(rosbag_name_list):
        print('Decompressing {}...'.format(rosbag_name.split('/')[-1]))
        bag = rosbag.Bag(rosbag_name, 'r')
        for topic, msg, ts in bag.read_messages():
            if topic == conf_dict.topic:
                image = decompress_image(msg, conf_dict.is_BGR)
                cv2.imwrite(str(output_dir / '{}.jpg'.format(ts)), image)
    
    print('Done')


if __name__ == '__main__':
    main()
