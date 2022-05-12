import os
import argparse
import numpy as np
import transformations
from easydict import EasyDict as edict
import cv2
import pandas as pd
import re
import shutil

from utils import create_dir_if_not_exist, load_yaml

file_dir = os.path.dirname(__file__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        type=str,
                        help='config file for preprocessing CMU dataset',
                        default=os.path.join(file_dir, 'configs/preprocess_config_cmu.yaml'))
    args = parser.parse_args()
    return args


def load_cmu_txt(cmu_txt_file):
    img_name_list = []
    img_pose_list = []
    with open(cmu_txt_file, 'r') as f:
        for line in f.readlines():
            line_split = line.strip().split(' ')
            img_name_list.append(line_split[0])
            line_split[1:] = map(float, line_split[1:])
            R = transformations.quaternion_matrix(np.array([line_split[1], line_split[2],
                                                            line_split[3], line_split[4]]))
            C = np.array([line_split[5], line_split[6], line_split[7]])
            t = -R[:3, :3] @ C
            R[:3, 3] = t
            img_pose_list.append(np.linalg.inv(R))

    return img_name_list, img_pose_list


def copy_images(input_dir, output_dir, downsampling_factor=1):
    for img_name in sorted(os.listdir(input_dir)):
        img = cv2.imread(os.path.join(input_dir, img_name))
        if downsampling_factor != 1:
            width = int(img.shape[1] / downsampling_factor)
            height = int(img.shape[0] / downsampling_factor)
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        else:
            resized_img = img

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
            cam_id = int(re.findall('\d+', img_name)[1]) + 1

            line = '%d %f %f %f %f %f %f %f %d %s\n' % (img_id, quat[0], quat[1], quat[2], quat[3],
                                                        xyz[0], xyz[1], xyz[2], cam_id, img_name)
            f.write(line)
            f.write('\n')


def write_points3D_txt(output_dir):
    with open(os.path.join(output_dir, 'points3D.txt'), 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        f.write('# Number of points: 0, mean track length: 0\n')


def write_cameras_txt(output_dir, cam0_conf, cam1_conf):
    with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: 1\n')

        # line = '%d %s %d %d %f %f %f %f %f %f %f %f\n' % \
        #     (1, 'OPENCV', 1440, 1080, 1039.13693, 1038.90465, 720.19014, 553.13684, -0.117387, 0.087465, 4.8e-5, 0.000289)
        line = '%d %s %d %d %f %f %f %f %f %f %f %f\n' % cam0_conf
        f.write(line)
        line = '%d %s %d %d %f %f %f %f %f %f %f %f\n' % cam1_conf
        f.write(line)


def main():
    args = parse_args()
    config_dict = edict(load_yaml(args.config))
    input_dir = config_dict['input_dir']

    # 1. check and create directories
    cmu_db_dir = os.path.join(input_dir, 'database')
    cmu_query_dir = os.path.join(input_dir, 'query')
    cmu_db_txt_file = os.path.join(input_dir, 'ground-truth-database-images-slice{}.txt'.format(config_dict.slice))
    cmu_query_txt_dir = os.path.join(input_dir, 'camera-poses')
    cmu_colmap_db_file_path = os.path.join(input_dir, 'database{}.db'.format(config_dict.slice))

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
    cam0_conf = [1]
    cam0_conf.extend(config_dict.cam0_conf)
    cam1_conf = [2]
    cam1_conf.extend(config_dict.cam1_conf)

    # scaling
    for i in range(2, 8):
        cam0_conf[i] = scaling_factor * cam0_conf[i]
        cam1_conf[i] = scaling_factor * cam1_conf[i]

    cam0_conf = tuple(cam0_conf)
    cam1_conf = tuple(cam1_conf)

    # 2. copy images and db file, load poses, etc.
    print('================ PREPROCESS CMU ================')
    print("Preprocessing %s ..." % input_dir)
    print("Output to %s ." % processed_dir)
    shutil.copyfile(cmu_colmap_db_file_path, os.path.join(sfm_colmap_dir, 'database.db'))
    copy_images(cmu_db_dir, processed_db_img_dir, downsampling_factor=config_dict['downsampling_factor'])
    copy_images(cmu_query_dir, processed_query_img_dir, downsampling_factor=config_dict['downsampling_factor'])

    img_name_list, img_pose_list = load_cmu_txt(cmu_db_txt_file)
    data = {
        'image_name': img_name_list,
        'image_pose': img_pose_list
    }
    df = pd.DataFrame(data, columns=data.keys())
    df.to_pickle(os.path.join(processed_db_dir, 'pose.pickle'))

    query_name_list, query_pose_list = [[], []]
    for file in os.listdir(cmu_query_txt_dir):
        file_path = os.path.join(cmu_query_txt_dir, file)
        _name_list, _pose_list = load_cmu_txt(file_path)
        query_name_list.extend(_name_list)
        query_pose_list.extend(_pose_list)
    query_data = {
        'image_name': query_name_list,
        'image_pose': query_pose_list
    }
    query_df = pd.DataFrame(query_data, columns=query_data.keys())
    query_df.to_pickle(os.path.join(processed_query_dir, 'pose.pickle'))

    # 3. perform 3D reconstruction (generate sfm_empty)
    print('================ Generating SfM Empty ================')
    print("Preprocessing %s ..." % processed_db_dir)
    print("Output to %s ." % output_dir)
    write_images_txt(sfm_empty_dir, img_name_list, img_pose_list)
    write_points3D_txt(sfm_empty_dir)
    write_cameras_txt(sfm_empty_dir, cam0_conf, cam1_conf)
    print('Done')


if __name__ == '__main__':
    main()
