import os
import argparse
from easydict import EasyDict as edict
import numpy as np
import cv2
import pandas as pd
import transformations

from utils import create_dir_if_not_exist, load_yaml
from read_write_model import read_model

file_dir = os.path.dirname(__file__)
db_dir_name = 'overcast-reference'
condition_list = ['dawn', 'dusk', 'night', 'night-rain', 'overcast-summer',
    'overcast-winter', 'rain', 'snow', 'sun']
position_list = ['left', 'right', 'rear']
position_id_map = {'left': 1, 'right': 2, 'rear': 3}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        type=str,
                        help='config file for preprocessing Robot Car dataset',
                        default=os.path.join(file_dir, 'configs/preprocess_config_robotcar.yaml'))
    args = parser.parse_args()
    return args


def copy_images(input_dir, output_dir, downsampling_factor=1, prefix=None):
    for img_name in sorted(os.listdir(input_dir)):
        img = cv2.imread(os.path.join(input_dir, img_name))
        if downsampling_factor != 1:
            width = int(img.shape[1] / downsampling_factor)
            height = int(img.shape[0] / downsampling_factor)
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        else:
            resized_img = img

        if prefix is not None:
            img_name = prefix + img_name

        cv2.imwrite(os.path.join(output_dir, img_name), resized_img)


def write_images_txt(output_dir, img_names, img_poses):
    with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write('# Number of images: 500, mean observations per image: 0\n')

        assert len(img_names) == len(img_poses)
        for idx in range(len(img_names)):
            img_id = idx
            img_pose_inv = np.linalg.inv(img_poses[idx])
            quat = transformations.quaternion_from_matrix(img_pose_inv)
            xyz = img_pose_inv[:3, 3]
            img_name = img_names[idx]
            cam_id = position_id_map[img_name.split('_')[0]]

            line = '%d %f %f %f %f %f %f %f %d %s\n' % (img_id, quat[0], quat[1], quat[2], quat[3],
                                                        xyz[0], xyz[1], xyz[2], cam_id, img_name)
            f.write(line)
            f.write('\n')


def write_points3D_txt(output_dir):
    with open(os.path.join(output_dir, 'points3D.txt'), 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        f.write('# Number of points: 0, mean track length: 0\n')


def write_cameras_txt(output_dir, cam_conf_dict):
    with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: 1\n')

        line = '%d %s %d %d %f %f %f %f %f %f %f %f\n' % cam_conf_dict['left']
        f.write(line)
        line = '%d %s %d %d %f %f %f %f %f %f %f %f\n' % cam_conf_dict['right']
        f.write(line)
        line = '%d %s %d %d %f %f %f %f %f %f %f %f\n' % cam_conf_dict['rear']
        f.write(line)


def main():
    args = parse_args()
    config_dict = edict(load_yaml(args.config))
    input_dir = config_dict['input_dir']

    # 1. check and create directories
    robotcar_img_dir = os.path.join(input_dir, 'images')
    robotcar_db_img_dir = os.path.join(robotcar_img_dir, db_dir_name)
    robotcar_query_img_dir = os.path.join(robotcar_img_dir, '{condition}')
    robotcar_db_sfm_dir = os.path.join(input_dir, '3D-models/sfm_sift')

    processed_dir = os.path.join(input_dir, 'processed')
    processed_db_dir = os.path.join(processed_dir, 'database')
    processed_db_img_dir = os.path.join(processed_db_dir, 'image')
    processed_query_dir = os.path.join(processed_dir, 'query')
    processed_query_img_dir = os.path.join(processed_query_dir, 'image')
    output_dir = os.path.join(processed_dir, 'output')
    sfm_colmap_dir = os.path.join(output_dir, 'sfm_colmap')
    sfm_empty_dir = os.path.join(output_dir, 'sfm_empty')
    evaluation_dir = os.path.join(processed_dir, 'evaluation')

    create_dir_if_not_exist(processed_dir)
    create_dir_if_not_exist(processed_dir)
    create_dir_if_not_exist(processed_db_dir)
    create_dir_if_not_exist(processed_db_img_dir)
    create_dir_if_not_exist(processed_query_dir)
    create_dir_if_not_exist(processed_query_img_dir)
    create_dir_if_not_exist(output_dir)
    create_dir_if_not_exist(sfm_colmap_dir)
    create_dir_if_not_exist(sfm_empty_dir)
    create_dir_if_not_exist(evaluation_dir)

    scaling_factor = 1 / config_dict['downsampling_factor']
    cam_conf_dict = {}
    for _, position in enumerate(position_list):
        cam_conf = [position_id_map[position]]
        cam_conf.extend(config_dict['{}_conf'.format(position)])

        for i in range(2, 8):
            cam_conf[i] = scaling_factor * cam_conf[i]

        cam_conf_dict[position] = tuple(cam_conf)

    # 2. copy images, load poses, etc
    print('================ PREPROCESS RobotCar ================')
    print("Preprocessing %s ..." % input_dir)
    print("Output to %s ." % processed_dir)
    for position in position_list:
        copy_images(os.path.join(robotcar_db_img_dir, position), processed_db_img_dir,
                    downsampling_factor=config_dict['downsampling_factor'],
                    prefix='{}_'.format(position))
    for condition in condition_list:
        for position in position_list:
            copy_images(os.path.join(robotcar_query_img_dir.format(condition=condition), position), processed_query_img_dir,
                        downsampling_factor=config_dict['downsampling_factor'],
                        prefix='{}_'.format(position))

    cameras, images, points3D = read_model(robotcar_db_sfm_dir, ext='.bin')
    img_name_list = []
    img_pose_list = []
    for image in images.values():
        R = transformations.quaternion_matrix(image.qvec)
        R[:3, 3] = image.tvec
        img_pose_list.append(np.linalg.inv(R))
        img_name_list.append('_'.join(image.name.split('/')[1:]))
    data = {
        'image_name': img_name_list,
        'image_pose': img_pose_list
    }
    df = pd.DataFrame(data, columns=data.keys())
    df.to_pickle(os.path.join(processed_db_dir, 'pose.pickle'))

    # 3. perform 3D reconstruction
    print('================ Generating SfM Empty ================')
    print("Preprocessing %s ..." % processed_db_dir)
    print("Output to %s ." % output_dir)
    write_images_txt(sfm_empty_dir, img_name_list, img_pose_list)
    write_points3D_txt(sfm_empty_dir)
    write_cameras_txt(sfm_empty_dir, cam_conf_dict)
    print('Done')


if __name__ == '__main__':
    main()
