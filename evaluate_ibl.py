import os
from pathlib import Path
import argparse
from easydict import EasyDict as edict
import numpy as np
import transformations
import pandas as pd

from utils import load_yaml, Plotter, create_dir_if_not_exist

file_dir = os.path.dirname(os.path.abspath(__file__))


def rotation_error(pose_error):
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error


def translation_error(pose_error):
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return trans_error, dx, dy, dz


def load_pose_txt(path, inverse=False):
    pose_dict = {}
    
    with open(path, 'r') as f:
        for line in f.readlines():
            line_list = line.strip('\n').split(' ')
            img_name = line_list[0]
            quat = np.array(list(map(float, line_list[1:5])))
            xyz = np.array(list(map(float, line_list[5:8])))

            pose = transformations.quaternion_matrix(quat)
            pose[:3, 3] = xyz

            if inverse:
                pose = np.linalg.inv(pose)
            
            pose_dict[img_name] = pose
    
    return pose_dict


def load_pose_pickle(path):
    pose_dict = {}

    df = pd.read_pickle(path)
    img_path_list = list(df.loc[:, 'image_name'])
    img_path_list = [path.split('/')[-1] for path in img_path_list]
    img_pose_list = np.array(list(df.loc[:, 'image_pose'].values))

    for idx, img_pose in enumerate(img_pose_list):
        pose_dict[img_path_list[idx]] = img_pose
    
    return pose_dict


def shift_origin(pose_dict, origin):
    for key in pose_dict.keys():
        pose_dict[key][:3, 3] -= origin


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        type=str,
                        help='config file for evaluation',
                        default=os.path.join(file_dir, 'evaluate_ibl_config.yaml'))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    conf_dict = edict(load_yaml(args.config))
    processed_dir = Path(conf_dict.processed_dir)
    evaluation_dir = processed_dir / 'evaluation'
    create_dir_if_not_exist(evaluation_dir)
    plotter = Plotter(evaluation_dir)

    result_path = processed_dir / 'output' / 'result.txt'
    query_pose_path = processed_dir / 'query' / 'pose.pickle'

    result_dict = load_pose_txt(result_path, inverse=True)
    query_dict = load_pose_pickle(query_pose_path)
    shift_origin(query_dict, np.array(conf_dict.origin))

    assert (len(result_dict) == len(query_dict))
    for img_name in result_dict.keys():
        assert (img_name in query_dict.keys())
    
    t_errs = []
    r_errs = []
    for img_name, result_pose in result_dict.items():
        query_pose = query_dict[img_name]
        pose_error = np.linalg.inv(result_pose).dot(query_pose)
        t_errs.append(translation_error(pose_error)[0])
        r_errs.append(rotation_error(pose_error))

    t_errs = np.array(t_errs)
    r_errs = np.array(r_errs)

    interval = 20
    # plot translational error
    t_counter = []
    t_min_thresh = 0
    # t_max_thresh = np.max(t_errs)
    t_max_thresh = 1
    t_thresh_list = np.linspace(t_min_thresh, t_max_thresh, interval)
    for thresh in t_thresh_list:
        t_counter.append(np.sum(t_errs < thresh) / len(t_errs))

    plotter.plot(([t_thresh_list, t_counter],), 'Distance threshold [meters]',
                 'Correctly localized queries [%]', 'Translational Error')    

    # plot rotational error
    r_counter = []
    r_min_thresh = 0
    r_max_thresh = np.max(r_errs)
    r_thresh_list = np.linspace(r_min_thresh, r_max_thresh, interval)
    for thresh in r_thresh_list:
        r_counter.append(np.sum(r_errs < thresh) / len(r_errs))
    
    plotter.plot(([r_thresh_list, r_counter],), 'Angle threshold [rad]',
                 'Correctly localized queries [%]', 'Rotational Error')


if __name__ == '__main__':
    main()
