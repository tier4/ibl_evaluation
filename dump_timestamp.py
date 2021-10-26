import os
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    folder = Path(args.folder)
    with open(folder.parent / 'timestamps.txt', 'w') as f:
        for file in sorted(os.listdir(folder)):
            f.write(file[:-4] + '\n')


if __name__ == '__main__':
    main()
