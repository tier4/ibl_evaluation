import argparse
from os import error
import numpy as np
import transformations
import pandas as pd
import matplotlib.pyplot as plt


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
    return trans_error


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--est_file', required=True)
    parser.add_argument('--gt_pickle', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load est poses, don't forget inverse
    est_poses = {}
    with open(args.est_file, 'r') as f:
        for line in f.readlines():
            filename = line.strip('\n').split(' ')[0]
            line = list(map(float, line.strip('\n').split(' ')[1:]))
            quat = np.array(line[0:4])
            translation = np.array(line[4:7])

            est_pose = transformations.quaternion_matrix(quat)
            est_pose[:3, 3] = translation

            est_pose = np.linalg.inv(est_pose)

            est_poses[filename] = est_pose
    
    # load gt poses
    df = pd.read_pickle(args.gt_pickle)
    image_paths = list(df.loc[:, 'image_path'])

    image_paths = [path.split('/')[-1] for path in image_paths]

    image_poses = np.array(list(df.loc[:, 'image_pose'].values))
    origin = np.array([65066.81680232, 664.20524287, 722.38443336])
    for idx in range(len(image_poses)):
        image_poses[idx, 0:3, 3] -= origin
    
    gt_poses = {}
    for idx, image_pose in enumerate(image_poses):
        gt_poses[image_paths[idx]] = image_pose
    
    # compare
    assert (len(gt_poses) == len(est_poses))
    for key, _ in est_poses.items():
        assert (key in gt_poses.keys())
    
    t_errs = []
    r_errs = []
    for key, est_pose in est_poses.items():
        gt_pose = gt_poses[key]
        pose_error = np.linalg.inv(est_pose).dot(gt_pose)
        t_errs.append(translation_error(pose_error))
        r_errs.append(rotation_error(pose_error))
    
    print(gt_poses['1608771075165282360.jpg'])
    print(est_poses['1608771075165282360.jpg'])
    print(gt_poses['1608771075165282360.jpg'][:3, 3] - est_poses['1608771075165282360.jpg'][:3, 3])

    t_errs = np.array(t_errs)
    r_errs = np.array(r_errs)

    # plot
    # t_threshold
    # t_threshold = np.linspace(0, 1, 11)
    t_threshold = np.linspace(0, 0.3, 11)
    # r_threshold
    
    count = []
    for t_thresh in t_threshold:
        count.append(np.sum(r_errs < t_thresh) / len(r_errs))
    
    print(count)

    plt.plot(t_threshold, count)
    plt.xlabel('threshold (deg)')
    plt.ylabel('Percentage (%)')
    plt.show()


if __name__ == '__main__':
    main()
