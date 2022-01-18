from pathlib import Path
import argparse
import cv2
import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help='the folder of images', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    output_dir = Path(args.input_dir).parent

    img_array = []
    for img_name in tqdm(sorted(glob.glob(args.input_dir + '/*.jpg'))[0:-1:4]):
        img = cv2.imread(img_name)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    out = cv2.VideoWriter(str(output_dir / 'video.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':
    main()
