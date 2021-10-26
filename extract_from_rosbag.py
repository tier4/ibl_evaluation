import os
import argparse
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
import rosbag
import rospy


def decompress_image(ros_image_compressed):
    try:
        np_arr = np.frombuffer(ros_image_compressed.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except BaseException as e:
        print(e)
    
    return image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bag')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    bag = rosbag.Bag(args.bag, 'r')
    # bags = sorted(os.listdir(Path(args.bag).parent))
    output_dir = Path(args.output)

    # for bag in bags:
        # bag = rosbag.Bag(str(Path(args.bag).parent / bag))
    for topic, msg, t in bag.read_messages():
        # if topic == '/sensing/camera/0/image_rect_color/compressed':
        #     image = decompress_image(msg)
        #     cv2.imwrite(str(output_dir / '{}.jpg'.format(t)), image)
        if topic == '/tf_static':
            print(msg)

    bag.close()


if __name__ == '__main__':
    main()
