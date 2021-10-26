import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    args = parser.parse_args()
    return args


def plot_poses(poses_xyz):
    fig = plt.figure(constrained_layout=True, figsize=(6,6))
    ax = plt.gca()
    ax.set_aspect('equal', 'datalim')
    font_size = 20

    plt.plot(poses_xyz[:, 0], poses_xyz[:, 1], '-o', markersize=1, color='b')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel('x (m)', fontsize=font_size)
    plt.ylabel('y (m)', fontsize=font_size, labelpad=-15)

    plt.show()


def main():
    args = parse_args()

    poses_xyz = []
    poses_quat = []
    timestamps = []
    with open(args.input, 'r') as file:
        yaml_str = ''
        for line in file.readlines():
            if line.strip() == '---':
                pose = yaml.full_load(yaml_str)
                xyz = np.array([pose['pose']['pose']['position']['x'],
                                pose['pose']['pose']['position']['y'],
                                pose['pose']['pose']['position']['z']])
                quat = np.array([pose['pose']['pose']['orientation']['x'],
                                 pose['pose']['pose']['orientation']['y'],
                                 pose['pose']['pose']['orientation']['z'],
                                 pose['pose']['pose']['orientation']['w']])
                secs = int(pose['header']['stamp']['secs']) * 1e9
                nsecs = int(pose['header']['stamp']['nsecs'])
                timestamp = secs + nsecs

                poses_xyz.append(xyz)
                poses_quat.append(quat)
                timestamps.append(timestamp)
                yaml_str = ''
            else:
                yaml_str += line
    
    poses_xyz = np.array(poses_xyz)
    poses_quat = np.array(poses_quat)
    timestamps = np.array(timestamps)

    plot_poses(poses_xyz)


if __name__ == '__main__':
    main()
