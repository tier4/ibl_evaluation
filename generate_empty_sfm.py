import os
import argparse
import numpy as np
import transformations
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    args  =parser.parse_args()
    return args


def main():
    args = parse_args()

    pd_path = os.path.join(args.input_dir, 'data.pickle')
    df = pd.read_pickle(pd_path)

    image_paths = list(df.loc[:, 'image_path'])
    image_poses = np.array(list(df.loc[:, 'image_pose'].values))

    # FIXME:
    # origin = np.copy(image_poses[0, 0:3, 3])
    # print('Origin: {}'.format(origin))
    origin = np.array([65066.81680232, 664.20524287, 722.38443336])
    for idx in range(len(image_poses)):
        image_poses[idx, 0:3, 3] -= origin

    # plot
    plt.figure(constrained_layout=True, figsize=(6,6))
    ax = plt.gca()
    ax.set_aspect('equal', 'datalim')
    font_size = 20

    plt.plot(image_poses[:, 0, 3], image_poses[:, 1, 3], '-o', markersize=1, color='b')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('x (m)', fontsize=font_size)
    plt.ylabel('y (m)', fontsize=font_size, labelpad=-15)

    # plt.show()

    # generate
    output_dir = os.path.join(args.input_dir, 'empty_sfm')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir, 'images.txt'), 'w') as file:
        file.write('# Image list with two lines of data per image:\n')
        file.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        file.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        file.write('# Number of images: 500, mean observations per image: 0\n')

        for idx, image_pose in enumerate(image_poses):
            image_id = idx
            image_pose_inv = np.linalg.inv(image_pose)
            quat = transformations.quaternion_from_matrix(image_pose_inv)
            translation = image_pose_inv[:3, 3]
            name = image_paths[idx].split('/')[-1]

            line = '%d %f %f %f %f %f %f %f %d %s\n' % (image_id, quat[0], quat[1], quat[2], quat[3], translation[0], translation[1], translation[2], 1, name)
            file.write(line)
            file.write('\n')

    with open(os.path.join(output_dir, 'points3D.txt'), 'w') as file:
        file.write('# 3D point list with one line of data per point:\n')
        file.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        file.write('# Number of points: 0, mean track length: 0\n')

    with open(os.path.join(output_dir, 'cameras.txt'), 'w') as file:
        file.write('# Camera list with one line of data per camera:\n')
        file.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        file.write('# Number of cameras: 1\n')
        
        line = '%d %s %d %d %f %f %f %f %f %f %f %f\n' % \
                (1, 'OPENCV', 1440, 1080, 1039.13693, 1038.90465, 720.19014, 553.13684, -0.117387, 0.087465, 4.8e-5, 0.000289)
        
        file.write(line)


if __name__ == '__main__':
    main()
