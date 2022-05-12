import os
from pathlib import Path
import argparse
import numpy as np
import cv2
from utils import Plotter
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', help='path to the image folder', type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    plotter = Plotter(output_dir='.', save_file=False)

    image_name_list = sorted(os.listdir(args.input_dir))
    print('Total images: {}.'.format(len(image_name_list)))

    prev_image = None
    cur_image = None
    image_diff = []
    for idx in range(len(image_name_list)):
        cur_image = cv2.imread(os.path.join(str(args.input_dir), image_name_list[idx]))
        cur_image = cur_image.astype(float)
        cur_image = cur_image / 255.0
        
        if prev_image is None:
            prev_image = cur_image
            continue
        else:
            image_diff.append(np.mean(np.abs(cur_image - prev_image)))
            prev_image = cur_image
    
    plotter.plot(([range(len(image_diff)), image_diff],), 'Idx', 'Diff.', 'Image Diff.')


if __name__ == '__main__':
    main()
