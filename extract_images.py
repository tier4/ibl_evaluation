import os
import argparse
from pathlib import Path

import numpy as np
import cv2
import rosbag
import rospy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bag', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--id', type=int, choices=[0,1,3,4,5], required=True)
    args = parser.parse_args()
    return args


def decompress_image(ros_image_compressed):
    try:
        np_arr = np.frombuffer(ros_image_compressed.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except BaseException as e:
        print(e)
    
    return image


def main():
    args = parse_args()
    
    topic_name = '/sensing/camera/{}/image_rect_color/compressed'.format(args.id)
    bag = rosbag.Bag(args.bag, 'r')
    output_dir = Path(args.output)

    for topic, msg, t in tqdm(bag.read_messages()):
        # if topic == topic_name:
        if topic == '/sensing/camera/traffic_light/left/image_raw/compressed':
            image = decompress_image(msg)
            cv2.imwrite(str(output_dir / '{}.jpg'.format(t)), image)
    
    bag.close()


if __name__ == '__main__':
    main()
