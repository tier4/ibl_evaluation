import os
from pathlib import Path
import argparse
import subprocess
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', help='path to the input folder', type=Path, required=True)
    parser.add_argument('-o', '--output_dir', help='path to the output folder', type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    for rosbag_name in tqdm(sorted(os.listdir(args.input_dir))):
        input_path = args.input_dir / rosbag_name
        output_name = '{:02d}.bag'.format(int(rosbag_name.split('_')[-1].split('.')[0]))
        output_path = args.output_dir / output_name

        command = 'bash filter_rosbag.sh %s %s' % (input_path, output_path)
        subprocess.call(command, shell=True, stdout=open(os.devnull, 'w'))


if __name__ == '__main__':
    main()
