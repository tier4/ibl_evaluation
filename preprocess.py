import os
import argparse
import numpy as np
import transformations

from utils import create_dir_if_not_exist, NDT2Image, load_yaml, Reconstructor

file_dir = os.path.dirname(__file__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        type=str,
                        help='config file for preprocessing',
                        default=os.path.join(file_dir, 'preprocess_config.yaml'))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config_dict = load_yaml(args.config)
    input_dir = config_dict['input_dir']

    # 1. check and create directories
    raw_dir = os.path.join(input_dir, 'raw')
    raw_db_img_dir = os.path.join(raw_dir, 'database/image')
    raw_query_img_dir = os.path.join(raw_dir, 'query/image')
    raw_db_pose_path = os.path.join(raw_dir, 'database/pose.txt')
    raw_query_pose_path = os.path.join(raw_dir, 'query/pose.txt')

    assert (os.path.exists(raw_dir))
    assert (os.path.exists(raw_db_img_dir))
    assert (os.path.exists(raw_query_img_dir))
    assert (os.path.exists(raw_db_pose_path))
    assert (os.path.exists(raw_query_pose_path))

    processed_dir = os.path.join(input_dir, 'processed')
    processed_db_dir = os.path.join(processed_dir, 'database')
    processed_db_img_dir = os.path.join(processed_db_dir, 'image')
    processed_query_dir = os.path.join(processed_dir, 'query')
    processed_query_img_dir = os.path.join(processed_query_dir, 'image')
    output_dir = os.path.join(processed_dir, 'output')
    evaluation_dir = os.path.join(processed_dir, 'evaluation')

    create_dir_if_not_exist(processed_dir)
    create_dir_if_not_exist(processed_db_dir)
    create_dir_if_not_exist(processed_db_img_dir)
    create_dir_if_not_exist(processed_query_dir)
    create_dir_if_not_exist(processed_query_img_dir)
    create_dir_if_not_exist(output_dir)
    create_dir_if_not_exist(evaluation_dir)

    T_sensor_cam = transformations.quaternion_matrix(config_dict.T_sensor_cam[:4])
    T_sensor_cam[:3, 3] = config_dict.T_sensor_cam[4:]
    cam_conf = [1]
    cam_conf = tuple(cam_conf.extend(config_dict.cam_conf))
    
    # 2. convert NDT poses to image poses, align timestamps, copy images, etc.
    db_processor = NDT2Image(os.path.join(raw_dir, 'database'),
                             processed_db_dir,
                             config_dict['db_range'], T_sensor_cam)
    db_processor.process()
    
    query_processeor = NDT2Image(os.path.join(raw_dir, 'query'),
                                 processed_query_dir,
                                 config_dict['query_range'], T_sensor_cam)
    query_processeor.process()

    # 3. perform 3D reconstruction (generate sfm_colmap and sfm_empty)
    reconstructor = Reconstructor(processed_db_dir,
                                  output_dir,
                                  np.array(config_dict['origin']),
                                  cam_conf)
    reconstructor.generate_sfm_empty()
    reconstructor.generate_sfm_colmap()


if __name__ == '__main__':
    main()
