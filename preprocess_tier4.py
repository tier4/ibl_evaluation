import os
import numpy as np
import transformations
import yaml
import pandas as pd
import argparse
import shutil

from se3 import interpolate_SE3


# T_repair = transformations.quaternion_matrix([0.5, 0.5, -0.5, 0.5])

# from optical to cam0
T_cam0_optical = transformations.quaternion_matrix([0.5, -0.5, 0.5, -0.5])
# from cam0 to sensor
T_sensor_cam0 = transformations.quaternion_matrix([0.9997767826301288, -0.005019424387927419, 0.0008972848758006599, 0.020503296623082125])
# from sensor to base
T_base_sensor = transformations.quaternion_matrix([0.9995149287258687, -0.00029495229864108036, -0.009995482472228997, -0.029494246683224673])

T_sensor_cam0[0:3, 3] = np.array([0.215, 0.031, -0.024])
T_base_sensor[0:3, 3] = np.array([0.6895, 0.0, 2.1])

T_cam0_base = np.linalg.inv(T_cam0_optical).dot(np.linalg.inv(T_sensor_cam0).dot(np.linalg.inv(T_base_sensor)))
T_base_cam0 = np.linalg.inv(T_cam0_base)


class Frame():
    def __init__(self, image_path, image_pose):
        self.image_path = image_path
        self.image_pose = image_pose


def find_timestamps_in_between(timestamp, timestamps_to_search):
    # ensure they are in between
    assert (timestamp >= timestamps_to_search[0])
    assert (timestamp <= timestamps_to_search[-1])

    index = 0
    while timestamps_to_search[index] <= timestamp:
        index += 1
    return index - 1, index


def interpolate(pose_i, pose_j, alpha):
    # pose
    pose_k = interpolate_SE3(pose_i, pose_j, alpha)

    return pose_k


def preprocess_tier4(seq_dir, output_dir, cam_subset_range):
    print("================ PREPROCESS Tier4 ================")
    print("Preprocessing %s" % seq_dir)
    print("Otput to %s" % output_dir)
    print("Camera images: %d => %d" % (cam_subset_range[0], cam_subset_range[1]))

    image_dir = os.path.join(seq_dir, 'image')

    # image timestamps
    image_timestamps = []
    for image_filename in sorted(os.listdir(image_dir)):
        image_timestamp = np.datetime64(int(image_filename[:-4]), 'ns')
        image_timestamps.append(image_timestamp)
    
    image_timestamps = np.array(image_timestamps)
    image_paths = [os.path.join(image_dir, p) for p in sorted(os.listdir(image_dir))]

    # load pose data
    ndt_poses = []
    ndt_timestamps = []
    with open(os.path.join(seq_dir, 'pose.txt'), 'r') as pose_file:
        yaml_str = ''
        for line in pose_file.readlines():
            if line.strip() == '---':
                pose = yaml.full_load(yaml_str)
                xyz = np.array([pose['pose']['pose']['position']['x'],
                                pose['pose']['pose']['position']['y'],
                                pose['pose']['pose']['position']['z']])
                quat = np.array([pose['pose']['pose']['orientation']['w'],
                                 pose['pose']['pose']['orientation']['x'],
                                 pose['pose']['pose']['orientation']['y'],
                                 pose['pose']['pose']['orientation']['z']])
                secs = int(int(pose['header']['stamp']['secs']) * 1e9)
                nsecs = int(pose['header']['stamp']['nsecs'])
                
                ndt_pose = transformations.quaternion_matrix(quat)
                ndt_pose[:3, 3] = xyz
                ndt_timestamp = np.datetime64(secs + nsecs, 'ns')

                ndt_poses.append(ndt_pose)
                ndt_timestamps.append(ndt_timestamp)

                yaml_str = ''
            else:
                yaml_str += line
        
    ndt_poses = np.array(ndt_poses)
    ndt_timestamps = np.array(ndt_timestamps)

    print('Read NDT poses.')

    # the first and last image timestamps must be between NDT timestamps
    assert (image_timestamps[cam_subset_range[0]] >= ndt_timestamps[0])
    assert (image_timestamps[cam_subset_range[1]] <= ndt_timestamps[-1])

    # take subset of interested images
    image_timestamps = image_timestamps[cam_subset_range[0] : cam_subset_range[1] + 1]
    image_paths = image_paths[cam_subset_range[0] : cam_subset_range[1] + 1]

    # for image_path in image_paths:
    #     src = image_path
    #     filename = src.split('/')[-1]
    #     dst_dir = '/home/zijiejiang/Documents/dataset/tier4/test/query'
    #     dst = os.path.join(dst_dir, filename)
    #     shutil.copyfile(src, dst)

    # convert to local time reference in seconds
    image_timestamps = (image_timestamps - ndt_timestamps[0]) / np.timedelta64(1, 's')
    ndt_timestamps = (ndt_timestamps - ndt_timestamps[0]) / np.timedelta64(1, 's')

    # interpolate
    frames = []
    for idx, image_timestamp in enumerate(image_timestamps):
        idx_i, idx_j = find_timestamps_in_between(image_timestamp, ndt_timestamps)
        t_i = ndt_timestamps[idx_i]
        t_j = ndt_timestamps[idx_j]
        alpha = (image_timestamp - t_i) / (t_j - t_i)
        image_pose = interpolate_SE3(ndt_poses[idx_i], ndt_poses[idx_j], alpha)
        image_pose = image_pose.dot(T_base_cam0)
        image_path = image_paths[idx]

        frame = Frame(image_path, image_pose)
        frames.append(frame)
    
    # save as pickle file
    data = {
        'image_path': [f.image_path for f in frames],
        'image_pose': [f.image_pose for f in frames]
    }

    df = pd.DataFrame(data, columns=data.keys())
    df.to_pickle(os.path.join(output_dir, 'data.pickle'))

    print('Saved pickle file.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('--mode', required=True, choices=['database', 'query'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    seq_dir = args.input_dir
    mode = args.mode
    output_dir = os.path.join('.', 'data', mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cam_subset_range = []
    if mode == 'database':
        # database
        image_start = '1605665633709718249.jpg'
        image_end = '1605665688600292370.jpg'
    else:
        # query
        image_start = '1608771020274660171.jpg'
        image_end = '1608771079569083400.jpg'
    image_list = sorted(os.listdir(os.path.join(seq_dir, 'image')))

    cam_subset_range = [image_list.index(image_start), image_list.index(image_end)]

    preprocess_tier4(seq_dir, output_dir, cam_subset_range)


if __name__ == '__main__':
    main()
