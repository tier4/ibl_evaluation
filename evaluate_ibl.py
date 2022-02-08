import os
from pathlib import Path
import argparse
from easydict import EasyDict as edict
import collections
import numpy as np
import transformations
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from utils import load_yaml, Plotter, create_dir_if_not_exist

file_dir = os.path.dirname(os.path.abspath(__file__))


def plot_evaluation(plotter, errs, interval, xlabel, ylabel, title, min_err=None, max_err=None):
    counter = []
    min_thresh = min_err if min_err else 0
    max_thresh = max_err if max_err else np.max(errs)
    thresh_list = np.linspace(min_thresh, max_thresh, interval)
    for thresh in thresh_list:
        counter.append(np.sum(errs < thresh) / len(errs) * 100)

    plotter.plot(([thresh_list, counter],), xlabel, ylabel, title)


def rotation_error(pose_error):
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error


def translation_error(pose_error):
    dx = np.abs(pose_error[0, 3])
    dy = np.abs(pose_error[1, 3])
    dz = np.abs(pose_error[2, 3])
    trans_error = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return trans_error, dx, dy, dz


def get_topK_error(img_name_list, error_type, error_list, output_dir, k=10):
    index_list = np.argsort(error_list)[-k:]

    with open(output_dir / 'top{}_error_{}.txt'.format(k, error_type), 'w') as f:
        nl = '\n'
        for idx in index_list:
            log_txt = 'image name : {}{}error_type : {}{}error : {}{}' \
                      .format(img_name_list[idx], nl, error_type, nl, error_list[idx], nl)

            f.write(log_txt + nl)


def get_inlier_percentage(result_log_dict, qname):
    qname = 'query/image/' + qname

    num_matches = result_log_dict['loc'][qname]['num_matches']
    num_inliers = result_log_dict['loc'][qname]['PnP_ret']['num_inliers']

    return num_inliers / num_matches


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
    result_log_path = processed_dir / 'output' / 'result_log.pkl'
    query_pose_path = processed_dir / 'query' / 'pose.pickle'

    result_dict = load_pose_txt(result_path, inverse=True)
    result_log_dict = pickle.load(open(result_log_path, 'rb'))
    query_dict = load_pose_pickle(query_pose_path)

    result_dict = collections.OrderedDict(sorted(result_dict.items()))
    query_dict = collections.OrderedDict(sorted(query_dict.items()))
    shift_origin(query_dict, np.array(conf_dict.origin))

    assert (len(result_dict) == len(query_dict))
    for img_name in result_dict.keys():
        assert (img_name in query_dict.keys())

    t_errs = []
    r_errs = []
    xt_errs = []
    yt_errs = []
    zt_errs = []
    inlier_percentage = []
    for img_name, result_pose in result_dict.items():
        query_pose = query_dict[img_name]
        pose_error = np.linalg.inv(result_pose).dot(query_pose)
        t_errs.append(translation_error(pose_error)[0])
        xt_errs.append(translation_error(pose_error)[1])
        yt_errs.append(translation_error(pose_error)[2])
        zt_errs.append(translation_error(pose_error)[3])
        r_errs.append(rotation_error(pose_error) * 180 / np.pi)
        inlier_percentage.append(100 * get_inlier_percentage(result_log_dict, img_name))

    t_errs = np.array(t_errs)
    xt_errs = np.array(xt_errs)
    yt_errs = np.array(yt_errs)
    zt_errs = np.array(zt_errs)
    r_errs = np.array(r_errs)

    interval = 20
    # plot translational error
    plot_evaluation(plotter, t_errs, interval, 'Distance threshold [meters]', 'Correctly localized queries [%]', 'Translational Error', max_err=1)
    plot_evaluation(plotter, xt_errs, interval, 'Distance threshold [meters]', 'Correctly localized queries [%]', 'X-Axis Translational Error', max_err=1)
    plot_evaluation(plotter, yt_errs, interval, 'Distance threshold [meters]', 'Correctly localized queries [%]', 'Y-Axis Translational Error', max_err=1)
    plot_evaluation(plotter, zt_errs, interval, 'Distance threshold [meters]', 'Correctly localized queries [%]', 'Z-Axis Translational Error', max_err=1)

    # plot rotational error
    plot_evaluation(plotter, r_errs, interval, 'Angle threshold [deg]', 'Correctly localized queries [%]', 'Rotational Error')

    get_topK_error(list(result_dict.keys()), 'translation', t_errs, evaluation_dir, k=20)
    get_topK_error(list(result_dict.keys()), 'rotation', r_errs, evaluation_dir, k=20)

    print('### Statics ###')
    print('# Rate (%) of correctly localized queries within 0.25m: {:.1f}%'.format(np.sum(t_errs < 0.25) / len(t_errs) * 100))
    print('# Rate (%) of correctly localized queries within 0.50m: {:.1f}%'.format(np.sum(t_errs < 0.50) / len(t_errs) * 100))
    print('# Rate (%) of correctly localized queries within 1.00m: {:.1f}%'.format(np.sum(t_errs < 1.00) / len(t_errs) * 100))

    print('# Median positional error: {:.2f} m'.format(np.median(t_errs)))
    print('# Median rotational error: {:.2f} degree'.format(np.median(r_errs)))


if __name__ == '__main__':
    main()
