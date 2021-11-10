import os
import time
import yaml
import pandas as pd
import shutil
import subprocess
import numpy as np
import transformations
import matplotlib.pyplot as plt

from se3 import interpolate_SE3


class NDT2Image:
    class Frame():
        def __init__(self, img_name, img_pose):
            self.img_name = img_name
            self.img_pose = img_pose

    def __init__(self, input_dir, output_dir, scope, img_ext='.jpg'):
        self.input_dir = input_dir
        self.input_img_dir = os.path.join(input_dir, 'image')
        self.ndt_pose_file = os.path.join(input_dir, 'pose.txt')

        self.output_dir = output_dir
        self.output_img_dir = os.path.join(output_dir, 'image')
        self.img_pose_file = os.path.join(output_dir, 'pose.pickle')
        
        input_img_names = sorted(os.listdir(self.input_img_dir))
        start_img = str(scope[0]) + img_ext
        end_img = str(scope[1]) + img_ext
        assert (start_img in input_img_names)
        assert (end_img in input_img_names)
        self.start_idx = input_img_names.index(start_img)
        self.end_idx = input_img_names.index(end_img)
        self.input_img_names = input_img_names[self.start_idx: self.end_idx+1]
    
        T_cam0_optical0 = transformations.quaternion_matrix([0.5, -0.5, 0.5, -0.5])
        T_sensor_cam0 = transformations.quaternion_matrix([0.9997767826301288, -0.005019424387927419, 0.0008972848758006599, 0.020503296623082125])
        T_base_sensor = transformations.quaternion_matrix([0.9995149287258687, -0.00029495229864108036, -0.009995482472228997, -0.029494246683224673])
        T_sensor_cam0[0:3, 3] = np.array([0.215, 0.031, -0.024])
        T_base_sensor[0:3, 3] = np.array([0.6895, 0.0, 2.1])

        self.T_base_optical0 = T_base_sensor.dot(T_sensor_cam0.dot(T_cam0_optical0))

    def name2ts(self, name):
        return int(name.split('.')[0])

    def load_ndt(self, ndt_pose_file):
        ndt_poses = []
        ndt_tss = []
        with open(ndt_pose_file, 'r') as f:
            yaml_str = ''
            for line in f.readlines():
                if line.strip() == '---':
                    ndt_dict = yaml.full_load(yaml_str)
                    xyz = np.array([ndt_dict['pose']['pose']['position']['x'],
                                    ndt_dict['pose']['pose']['position']['y'],
                                    ndt_dict['pose']['pose']['position']['z']])
                    quat = np.array([ndt_dict['pose']['pose']['orientation']['w'],
                                     ndt_dict['pose']['pose']['orientation']['x'],
                                     ndt_dict['pose']['pose']['orientation']['y'],
                                     ndt_dict['pose']['pose']['orientation']['z']])
                    sec = int(int(ndt_dict['header']['stamp']['secs']) * 1e9)
                    nsec = int(ndt_dict['header']['stamp']['nsecs'])

                    ndt_pose = transformations.quaternion_matrix(quat)
                    ndt_pose[:3, 3] = xyz
                    ndt_ts = np.datetime64(sec + nsec, 'ns')

                    ndt_poses.append(ndt_pose)
                    ndt_tss.append(ndt_ts)

                    yaml_str = ''
                else:
                    yaml_str += line
        
        return ndt_poses, np.array(ndt_tss)

    def find_timestamp_in_between(self, ts, tss_to_search):
        assert (ts >= tss_to_search[0])
        assert (ts <= tss_to_search[-1])

        index = 0
        while tss_to_search[index] <= ts:
            index += 1
        
        return index - 1, index

    def interpolate_pose(self, pose_i, pose_j, alpha):
        pose_k = interpolate_SE3(pose_i, pose_j, alpha)

        return pose_k

    def copy_images(self, input_dir, output_dir, input_img_names):
        for img_name in input_img_names:
            shutil.copyfile(os.path.join(input_dir, img_name), os.path.join(output_dir, img_name))

    def process(self):
        print('================ PREPROCESS Tier4 ================')
        print("Preprocessing %s ..." % self.input_dir)
        print("Output to %s ." % self.output_dir)
        print("Camera images: %d => %d ." % (self.start_idx, self.end_idx))

        img_names = self.input_img_names
        img_tss = np.array([np.datetime64(self.name2ts(name), 'ns') for name in img_names])
        ndt_poses, ndt_tss = self.load_ndt(self.ndt_pose_file)

        img_tss = (img_tss - ndt_tss[0]) / np.timedelta64(1, 's')
        ndt_tss = (ndt_tss - ndt_tss[0]) / np.timedelta64(1, 's')

        frames = []
        for idx, img_ts in enumerate(img_tss):
            idx_i, idx_j = self.find_timestamp_in_between(img_ts, ndt_tss)
            t_i = ndt_tss[idx_i]
            t_j = ndt_tss[idx_j]
            alpha = (img_ts - t_i) / (t_j - t_i)
            img_pose = interpolate_SE3(ndt_poses[idx_i], ndt_poses[idx_j], alpha)
            img_pose = img_pose.dot(self.T_base_optical0)
            img_name = img_names[idx]

            frame = self.Frame(img_name, img_pose)
            frames.append(frame)
        
        data = {
            'image_name': [f.img_name for f in frames],
            'image_pose': [f.img_pose for f in frames]
        }

        df = pd.DataFrame(data, columns=data.keys())
        df.to_pickle(self.img_pose_file)

        print('Pickle saved.')

        self.copy_images(self.input_img_dir,
                         self.output_img_dir,
                         self.input_img_names)

        print('Done.')


class Reconstructor:
    def __init__(self, db_dir, output_dir, origin):
        self.db_dir = db_dir
        self.db_img_dir = os.path.join(db_dir, 'image')
        self.img_pose_file = os.path.join(db_dir, 'pose.pickle')

        self.output_dir = output_dir
        self.sfm_colmap_dir = os.path.join(output_dir, 'sfm_colmap')
        self.sfm_empty_dir = os.path.join(output_dir, 'sfm_empty')
        create_dir_if_not_exist(self.sfm_colmap_dir)
        create_dir_if_not_exist(self.sfm_empty_dir)

        self.origin = origin

    def shift_origin(self, poses, origin):
        for idx in range(len(poses)):
            poses[idx][0:3, 3] -= origin
        
        return poses
    
    def write_images_txt(self, output_dir, img_names, img_poses):
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

                line = '%d %f %f %f %f %f %f %f %d %s\n' % (img_id, quat[0], quat[1], quat[2], quat[3], \
                                                            xyz[0], xyz[1], xyz[2], 1, img_name)
                f.write(line)
                f.write('\n')

    def write_points3D_txt(self, output_dir):
        with open(os.path.join(output_dir, 'points3D.txt'), 'w') as f:
            f.write('# 3D point list with one line of data per point:\n')
            f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
            f.write('# Number of points: 0, mean track length: 0\n')

    # FIXME: Camera params should be given in the config file.
    def write_cameras_txt(self, output_dir):
        with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
            f.write('# Camera list with one line of data per camera:\n')
            f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
            f.write('# Number of cameras: 1\n')

            line = '%d %s %d %d %f %f %f %f %f %f %f %f\n' % \
                (1, 'OPENCV', 1440, 1080, 1039.13693, 1038.90465, 720.19014, 553.13684, -0.117387, 0.087465, 4.8e-5, 0.000289)
            f.write(line)

    def generate_sfm_empty(self):
        print('================ Generating SfM Empty ================')
        print("Preprocessing %s ..." % self.db_dir)
        print("Output to %s ." % self.output_dir)
        df = load_pickle(self.img_pose_file)

        img_names = list(df.loc[:, 'image_name'])
        img_poses = list(df.loc[:, 'image_pose'].values)
        img_poses = self.shift_origin(img_poses, self.origin)

        self.write_images_txt(self.sfm_empty_dir, img_names, img_poses)
        self.write_points3D_txt(self.sfm_empty_dir)
        self.write_cameras_txt(self.sfm_empty_dir)

        print('Done')
        
    def generate_sfm_colmap(self):
        print('================ Generating SfM Colmap ================')
        print("Preprocessing %s ..." % self.db_dir)
        print("Output to %s ." % self.output_dir)

        print('Running feature_extractor...')
        timer = time.time()
        command = 'colmap feature_extractor --database_path %s  --image_path %s --ImageReader.camera_model %s --ImageReader.camera_params %s' \
                  % (os.path.join(self.sfm_colmap_dir, 'database.db'),
                     self.db_img_dir,
                     'OPENCV',
                     ','.join(list(map(str, [1039.13693,1038.90465,720.19014,553.13684,-0.117387,0.087465,4.8e-5,0.000289]))))
        subprocess.call(command, shell=True, stdout=open(os.devnull, 'w'))
        print('Cost %.2f s.' % (time.time() - timer))

        print('Running matcher ...')
        timer = time.time()
        # FIXME: Overlap etc. should be given in the config file.
        command = 'colmap sequential_matcher --database_path %s \
                   --SequentialMatching.overlap %s \
                   --SequentialMatching.quadratic_overlap %s' \
                  % (os.path.join(self.sfm_colmap_dir, 'database.db'),
                     20,
                     0)
        subprocess.call(command, shell=True, stdout=open(os.devnull, 'w'))
        print('Cost %.2f s.' % (time.time() - timer))

        print('Running mapper ...')
        timer = time.time()
        command = 'colmap mapper --database_path %s \
                   --image_path %s \
                   --output_path %s' \
                  % (os.path.join(self.sfm_colmap_dir, 'database.db'),
                     self.db_img_dir,
                     self.sfm_colmap_dir)
        subprocess.call(command, shell=True, stdout=open(os.devnull, 'w'))
        print('Cost %.2f s.' % (time.time() - timer))


class Plotter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.count = 0
    
    def plot(self, plots, xlabel, ylabel, title, labels=None, equal_axes=False, filename=None, callback=None, colors=None):
        if not labels:
            labels_txt = [None] * len(plots)
        else:
            labels_txt = labels
        
        assert (len(plots) == len(labels_txt))

        plt.clf()
        
        for i in range(0, len(plt.plot)):
            args = {
                'label': labels_txt[i]
            }
            if colors:
                args['color'] = colors[i]
            
            plt.plot(*plots[i], linewidth=1, **args)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        if equal_axes:
            plt.axis('equal')
        
        if labels:
            plt.legend()
        
        plt.grid()
        
        if filename is None:
            filename = '%02d_%s.png' % (self.count, '_'.join(title.lower().split()))
        
        if callback is not None:
            callback(plt.gcf(), plt.gca())
        
        plt.savefig(os.path.join(self.output_dir, filename), format='png', bbox_inches='tight', pad_inches=0)
        self.count += 1


def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_yaml(path):
    with open(path, 'r') as f:
        yaml_dict = yaml.full_load(f)
    
    return yaml_dict


def load_pickle(path):
    df =  pd.read_pickle(path)

    return df
