import os
import time
import yaml
import pandas as pd
import shutil
import subprocess
import numpy as np
import open3d as o3d
import cv2
import transformations
import collections
import png
import matplotlib.pyplot as plt

from se3 import interpolate_SE3


class NDT2Image:
    class Frame():
        def __init__(self, img_name, img_pose, pcd_name=None, T_img_pcd=None):
            self.img_name = img_name
            self.img_pose = img_pose
            self.pcd_name = pcd_name
            self.T_img_pcd = T_img_pcd

    def __init__(self, input_dir, output_dir, scope, T_sensor_cam, downsampling_factor, cropping_param, img_ext='.jpg', have_pcd=False):
        self.input_dir = input_dir
        self.input_img_dir = os.path.join(input_dir, 'image')
        self.input_pcd_dir = os.path.join(input_dir, 'pcd')
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

        self.have_pcd = have_pcd
        if self.have_pcd:
            self.input_pcd_names = sorted(os.listdir(self.input_pcd_dir))

        T_cam_optical = transformations.quaternion_matrix([0.5, -0.5, 0.5, -0.5])
        # T_sensor_cam = transformations.quaternion_matrix([0.9997767826301288, -0.005019424387927419, 0.0008972848758006599, 0.020503296623082125])
        T_base_sensor = transformations.quaternion_matrix([0.9995149287258687, -0.00029495229864108036, -0.009995482472228997, -0.029494246683224673])
        # T_sensor_cam[0:3, 3] = np.array([0.215, 0.031, -0.024])
        T_base_sensor[0:3, 3] = np.array([0.6895, 0.0, 2.1])

        self.T_base_optical = T_base_sensor.dot(T_sensor_cam.dot(T_cam_optical))
        self.downsampling_factor = downsampling_factor
        self.x0 = cropping_param[0]
        self.y0 = cropping_param[1]
        self.x1 = cropping_param[2]
        self.y1 = cropping_param[3]

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
    
    def find_timestamp_nearest(self, ts, tss_to_search):
        try:
            assert (ts >= tss_to_search[0])
            assert (ts <= tss_to_search[-1])
        except AssertionError as e:
            print(ts)
            print(tss_to_search[0])
            print(tss_to_search[-1])
            exit(0)

        return np.argmin(np.abs(tss_to_search - ts))

    def interpolate_pose(self, pose_i, pose_j, alpha):
        pose_k = interpolate_SE3(pose_i, pose_j, alpha)

        return pose_k

    def copy_images(self, input_dir, output_dir, input_img_names, downsampling_factor, threshold=0.01):
        prev_image_dict = None
        cur_image_dict = None
        ret_img_name_list = []

        for img_name in input_img_names:
            cur_image = cv2.imread(os.path.join(input_dir, img_name))
            width = int(cur_image.shape[1] / downsampling_factor)
            height = int(cur_image.shape[0] / downsampling_factor)
            resized_img = cv2.resize(cur_image, (width, height), interpolation=cv2.INTER_LANCZOS4)
            
            cur_image_dict = {
                'name': img_name,
                'image': resized_img
            }

            if prev_image_dict is None:
                prev_image_dict = cur_image_dict
                continue
            else:
                if np.mean(np.abs(cur_image_dict['image'].astype(float) / 255.0 - \
                                  prev_image_dict['image'].astype(float) / 255.0)) > threshold:
                    image_to_write = prev_image_dict['image'][self.y0:self.y1, self.x0:self.x1]
                    cv2.imwrite(os.path.join(output_dir, prev_image_dict['name']), image_to_write)
                    ret_img_name_list.append(prev_image_dict['name'])
                    prev_image_dict = cur_image_dict
            
        print("Dynamic images: %d ..." % len(ret_img_name_list))
        return ret_img_name_list

    def process(self):
        print('================ PREPROCESS Tier4 ================')
        print("Preprocessing %s ..." % self.input_dir)
        print("Output to %s ." % self.output_dir)
        print("Camera images: %d => %d ." % (self.start_idx, self.end_idx))

        img_names = self.copy_images(self.input_img_dir,
                                     self.output_img_dir,
                                     self.input_img_names,
                                     self.downsampling_factor)

        img_tss = np.array([np.datetime64(self.name2ts(name), 'ns') for name in img_names])
        ndt_poses, ndt_tss = self.load_ndt(self.ndt_pose_file)
        start_ts = ndt_tss[0]

        img_tss = (img_tss - start_ts) / np.timedelta64(1, 's')
        ndt_tss = (ndt_tss - start_ts) / np.timedelta64(1, 's')
        if self.have_pcd:
            pcd_tss = np.array([np.datetime64(self.name2ts(name) * 1000, 'ns') for name in self.input_pcd_names]) # IMPORTANT: * 1000
            pcd_tss = (pcd_tss - start_ts) / np.timedelta64(1, 's')

        frames = []
        for idx, img_ts in enumerate(img_tss):
            idx_i, idx_j = self.find_timestamp_in_between(img_ts, ndt_tss)
            t_i = ndt_tss[idx_i]
            t_j = ndt_tss[idx_j]
            alpha = (img_ts - t_i) / (t_j - t_i)
            img_pose = interpolate_SE3(ndt_poses[idx_i], ndt_poses[idx_j], alpha)
            img_pose = img_pose.dot(self.T_base_optical)
            img_name = img_names[idx]

            if self.have_pcd:
                pcd_idx_nearest = self.find_timestamp_nearest(img_ts, pcd_tss)
                pcd_t_nearest = pcd_tss[pcd_idx_nearest]

                # GT. img pose at pcd timestamp
                pcd_idx_i, pcd_idx_j = self.find_timestamp_in_between(pcd_t_nearest, ndt_tss)
                pcd_t_i = ndt_tss[idx_i]
                pcd_t_j = ndt_tss[idx_j]
                pcd_alpha = (img_ts - pcd_t_i) / (pcd_t_j - pcd_t_i)
                pcd_img_pose = interpolate_SE3(ndt_poses[pcd_idx_i], ndt_poses[pcd_idx_j], pcd_alpha)
                pcd_img_pose = pcd_img_pose.dot(self.T_base_optical)

                T_img_pcd = np.linalg.inv(img_pose).dot(pcd_img_pose)
                pcd_name = self.input_pcd_names[pcd_idx_nearest]

                frame = self.Frame(img_name, img_pose, pcd_name, T_img_pcd)
            else:
                frame = self.Frame(img_name, img_pose)
            
            frames.append(frame)
        
        data = {
            'image_name': [f.img_name for f in frames],
            'image_pose': [f.img_pose for f in frames],
            'pcd_name': [f.pcd_name for f in frames],
            'T_img_pcd': [f.T_img_pcd for f in frames]
        }

        df = pd.DataFrame(data, columns=data.keys())
        df.to_pickle(self.img_pose_file)

        print('Pickle saved.')

        print('Done.')


class Reconstructor:
    def __init__(self, db_dir, output_dir, origin, cam_conf):
        self.db_dir = db_dir
        self.db_img_dir = os.path.join(db_dir, 'image')
        self.img_pose_file = os.path.join(db_dir, 'pose.pickle')

        self.output_dir = output_dir
        self.sfm_colmap_dir = os.path.join(output_dir, 'sfm_colmap')
        self.sfm_empty_dir = os.path.join(output_dir, 'sfm_empty')
        create_dir_if_not_exist(self.sfm_colmap_dir)
        create_dir_if_not_exist(self.sfm_empty_dir)

        self.origin = origin
        self.cam_conf = cam_conf

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

            # line = '%d %s %d %d %f %f %f %f %f %f %f %f\n' % \
            #     (1, 'OPENCV', 1440, 1080, 1039.13693, 1038.90465, 720.19014, 553.13684, -0.117387, 0.087465, 4.8e-5, 0.000289)
            line = '%d %s %d %d %f %f %f %f %f %f %f %f\n' % self.cam_conf
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
        print('cam param:')
        print(self.cam_conf[4:])
        command = 'colmap feature_extractor --database_path %s  --image_path %s --ImageReader.camera_model %s --ImageReader.camera_params %s' \
                  % (os.path.join(self.sfm_colmap_dir, 'database.db'),
                     self.db_img_dir,
                     'OPENCV',
                     ','.join(list(map(str, self.cam_conf[4:]))))
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

        # print('Running mapper ...')
        # timer = time.time()
        # command = 'colmap mapper --database_path %s \
        #            --image_path %s \
        #            --output_path %s' \
        #           % (os.path.join(self.sfm_colmap_dir, 'database.db'),
        #              self.db_img_dir,
        #              self.sfm_colmap_dir)
        # subprocess.call(command, shell=True, stdout=open(os.devnull, 'w'))
        # print('Cost %.2f s.' % (time.time() - timer))


class DepthCompleter:
    def __init__(self, db_dir, pcd_dir, output_dir, T_sensor_cam, cam_conf):
        self.db_dir = db_dir
        self.db_img_dir = os.path.join(db_dir, 'image')
        self.img_pose_file = os.path.join(db_dir, 'pose.pickle')

        self.pcd_dir = pcd_dir

        self.output_dir = output_dir
        self.depth_dir = os.path.join(output_dir, 'depth')
        self.depth_visualization_dir = os.path.join(output_dir, 'depth_visualization')
        create_dir_if_not_exist(self.depth_dir)
        create_dir_if_not_exist(self.depth_visualization_dir)

        T_cam_optical = transformations.quaternion_matrix([0.5, -0.5, 0.5, -0.5])
        T_base_sensor = transformations.quaternion_matrix([0.9995149287258687, -0.00029495229864108036, -0.009995482472228997, -0.029494246683224673])
        T_base_sensor[0:3, 3] = np.array([0.6895, 0.0, 2.1])

        self.T_base_optical = T_base_sensor.dot(T_sensor_cam.dot(T_cam_optical))
        self.T_optical_base = np.linalg.inv(self.T_base_optical)
        self.cam_conf = cam_conf

        self.W = cam_conf[2]
        self.H = cam_conf[3]
        self.FX = cam_conf[4]
        self.FY = cam_conf[5]
        self.X0 = cam_conf[6]
        self.Y0 = cam_conf[7]
        self.K1 = cam_conf[8]
        self.K2 = cam_conf[9]
    
    def process(self):
        print('================ Processing PCD Files ================')
        print("Preprocessing %s ..." % self.pcd_dir)
        print("Output to %s ." % self.depth_dir)

        df = load_pickle(self.img_pose_file)
        pcd_names = list(df.loc[:, 'pcd_name'])
        img_names = list(df.loc[:, 'image_name'])
        T_img_pcd_list = list(df.loc[:, 'T_img_pcd'].values)

        for idx in range(len(pcd_names)):
            pcd_path = os.path.join(self.pcd_dir, pcd_names[idx])
            T_img_pcd = T_img_pcd_list[idx]

            pcd = o3d.io.read_point_cloud(pcd_path)

            pcd_points = np.asarray(pcd.points)
            pcd_points = np.hstack([pcd_points, np.ones((len(pcd_points), 1))])
            cam_points = self.T_optical_base @ np.transpose(pcd_points)

            cam_points = T_img_pcd @ cam_points # Timestamp Alignment

            cam_points = np.transpose(cam_points)[:, :3]

            xyz_view = get_camera_view_pointcloud(cam_points, self.H, self.W, self.K1, self.K2, self.FX, self.FY, self.X0, self.Y0)
            occluded_inds = get_occluded_points(xyz_view, 0.002, 0.08)
            occluded_inds = set(occluded_inds)
            visible_indices = [
                i for i in range(xyz_view.shape[0]) if i not in occluded_inds
            ]
            visible_xyz = xyz_view[visible_indices, :]
            visible_xyz = filter_out_ot_frame_points(visible_xyz, self.H, self.W, self.K1, self.K2, self.FX, self.FY, self.X0, self.Y0)

            inds_to_keep = sample_uniform(visible_xyz, 1e-3)
            visible_xyz = visible_xyz[inds_to_keep]

            z_image = render_z(visible_xyz, self.H, self.W, self.K1, self.K2, self.FX, self.FY, self.X0, self.Y0)
            depth_image = (z_image * 256).astype(np.uint16)

            image_jet = cv2.applyColorMap(
                    np.uint8(z_image / np.amax(z_image) * 255),
                    cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.depth_visualization_dir, img_names[idx][:-4] + '.png'), image_jet)

            depth_file_path = os.path.join(self.depth_dir, img_names[idx][:-4] + '.png')
            with open(depth_file_path, 'wb') as f:
                # pypng is used because cv2 cannot save uint16 format images
                writer = png.Writer(width=depth_image.shape[1],
                                    height=depth_image.shape[0],
                                    bitdepth=16,
                                    greyscale=True)
                writer.write(f, depth_image)


class Plotter:
    def __init__(self, output_dir, save_file=True):
        self.output_dir = output_dir
        self.count = 0
        self.save_file = save_file
    
    def plot(self, plots, xlabel, ylabel, title, labels=None, equal_axes=False, filename=None, callback=None, colors=None):
        if not labels:
            labels_txt = [None] * len(plots)
        else:
            labels_txt = labels
        assert (len(plots) == len(labels_txt))

        plt.clf()
        for i in range(0, len(plots)):
            args = {
                'label': labels_txt[i]
            }
            if colors:
                args['color'] = colors[i]
            plt.plot(*plots[i], linewidth=1, **args)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)

        if equal_axes:
            plt.axis('equal')
        
        if labels:
            plt.legend()
        
        plt.grid()
        if filename is None:
            filename = '%02d_%s.png' % (self.count, '_'.join(title.lower().split()))
        
        if callback is not None:
            callback(plt.gcf(), plt.gca())
        
        if self.save_file:
            plt.savefig(os.path.join(self.output_dir, filename), format='png', bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        
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


def get_camera_view_pointcloud(xyz,
                               H, W, K1, K2, FX, FY, X0, Y0):
    """Prune points outside of the view.
    Args:
        xyz: A Nx3 matrix, point cloud in homogeneous coordinates. The k-th row
        is (x, y, z), where x, y, z are the coordinates of the k-th point.
    Returns:
        Mx3 (M < N) matrix representing the point cloud in the camera view.

        Only points that fall within the camera viweing angle and are in front of
        the camera are kept.
    """
    x, y, z = _split(xyz)
    u, v = _project_and_distort(x, y, z, K1, K2, FX, FY, X0, Y0)
    # Remove points that are out of frame. Keep some margin (1.05), to make sure
    # occlusions are addressed correctly at the edges of the field of view. For
    # example a point that is just slightly out of frame can occlude a neighboring
    # point inside the frame.
    valid_mask = np.logical_and.reduce(
        (z > 0.0, u > -0.05 * W, u < W * 1.05, v > -0.05 * H, v < H * 1.05),
        axis=0)
    valid_points = valid_mask.nonzero()[0]
    return xyz[valid_points, :]


def get_occluded_points(xyz, neighborhood_radius, z_threshold):
  """Remove points that are occluded by others from a camera-view point cloud.
  Args:
    xyz: A Nx3 matrix representing the point cloud in the camera view.
    neighborhood_radius: The radius around each point in which it occludes
      others.
    z_threshold: Minimum z distance betweem two points for them considered to
      be occluding each other. If two points are verty close in z, they likely
      belong to the same surface and thus do not occlude each other.
  Returns:
    A list of indices in xyz corresponding to points that are occluded.
  """

  def get_bin(xz, yz):
    xbin = int(round(xz / neighborhood_radius))
    ybin = int(round(yz / neighborhood_radius))
    return xbin, ybin

  xs, ys, zs = _split(xyz)
  xzs = xs / zs
  yzs = ys / zs
  grid = collections.defaultdict(lambda: np.inf)
  for ind in range(xyz.shape[0]):
    # Place each point in the bin where it belongs, and in the neighboring bins.
    # Keep only the closest point to the camera in each bin.
    xbin, ybin = get_bin(xzs[ind], yzs[ind])
    for i in range(-1, 2):
      for j in range(-1, 2):
        grid[(xbin + i, ybin + j)] = min(grid[(xbin + i, ybin + j)], zs[ind])

  occluded_indices = []
  for ind in range(xyz.shape[0]):
    # Loop over all points and see if they are occluded, by finding the closest
    # point to the camera within the same bin and testing for the occlusion
    # condition. A point is occluded if there is another point in the same bin
    # that is far enough in z, so that it cannot belong to the same surface,
    zmin = grid[get_bin(xzs[ind], yzs[ind])]
    if zmin < (1 - z_threshold) * zs[ind]:
      occluded_indices.append(ind)
  return occluded_indices


def render_rgb(xyz, c,
               H, W, K1, K2, FX, FY, X0, Y0):
  """Given a colored cloud in camera coordinates, render an image.
  This function is useful for visualization / debugging.
  Args:
    xyz: A 3xN matrix representing the point cloud in the camera view.
    c: A N-long vector containing (greyscale) colors of the points.
  Returns:
    A rendered image.
  """
  x, y, z = _split(xyz)
  u, v = _project_and_distort(x, y, z, K1, K2, FX, FY, X0, Y0)
  u = np.floor(u).astype(int)
  v = np.floor(v).astype(int)

  rendered_c = np.full((int(H), int(W)), 0.0)
  rendered_c[v, u] = c
  rendered_c = np.stack([rendered_c] * 3, axis=2)
  return rendered_c


def render_z(xyz,
             H, W, K1, K2, FX, FY, X0, Y0):
  """Given a colored cloud in camera coordinates, render a depth map.
  This function is useful for visualization / debugging.
  Args:
    xyz: A 3xN matrix representing the point cloud in the camera view.
  Returns:
    A rendered depth map.
  """
  x, y, z = _split(xyz)
  u, v = _project_and_distort(x, y, z, K1, K2, FX, FY, X0, Y0)
  u = np.floor(u).astype(int)
  v = np.floor(v).astype(int)
  rendered_z = np.full((int(H), int(W)), -np.inf)
  rendered_z[v, u] = z
  rendered_z = np.where(rendered_z == -np.inf, 0, rendered_z)
  return rendered_z


def filter_out_ot_frame_points(xyz,
                               H, W, K1, K2, FX, FY, X0, Y0):
  """Remove all points in a camera-view pointcloud that are out of frame.
  Args:
    xyz: A Nx3 matrix representing the point cloud in the camera view.
  Returns:
    A Mx3 matrix and a M-long vector representing the filtered colored point
    cloud.
  """
  x, y, z = _split(xyz)
  u, v = _project_and_distort(x, y, z, K1, K2, FX, FY, X0, Y0)
  u = np.floor(u).astype(int)
  v = np.floor(v).astype(int)
  valid_mask = np.logical_and.reduce((u >= 0, u < W, v >= 0, v < H), axis=0)
  valid_points = valid_mask.nonzero()[0]
  return xyz[valid_points, :]


def sample_uniform(xyz, bin_size):
  """subsamples a point cloud to be more uniform in perspective coordinates.
  Args:
    xyz: A Nx3 matrix representing the point cloud in the camera view.
    bin_size: Size of a square in which we allow only a single point.
  Returns:
    A list of indices, corresponding to a subset of the original `xyz`, to keep.
  """
  x, y, z = _split(xyz)
  xbins = (x / z / bin_size)
  ybins = (y / z / bin_size)
  xbins_rounded = np.round(xbins)
  ybins_rounded = np.round(ybins)
  xbins_diff = xbins_rounded - xbins
  ybins_diff = ybins_rounded - ybins
  diff_sq = xbins_diff**2 + ybins_diff**2

  bin_to_ind = {}
  for ind in range(len(diff_sq)):
    bin_ = (xbins_rounded[ind], ybins_rounded[ind])
    if bin_ not in bin_to_ind or diff_sq[ind] < bin_to_ind[bin_][1]:
      bin_to_ind[bin_] = (ind, diff_sq[ind])

  inds_to_keep = sorted([i[0] for i in bin_to_ind.values()])
  return inds_to_keep


def _split(matrix):
  return [
      np.squeeze(v, axis=1) for v in np.split(matrix, matrix.shape[1], axis=1)
  ]


def _project_and_distort(x, y, z,
                         K1, K2, FX, FY, X0, Y0):
    """Apply perspective projection and distortion on a point cloud.
    Args:
        x: A vector containing the x coordinates of the points.
        y: A vector containing the y coordinates of the points, same length as x.
        z: A vector containing the z coordinates of the points, same length as x.
    Returns:
        A tuple of two vectors of the same length as x, containing the image-plane
        coordinates (u, v) of the point cloud.
    """
    xz = (x / z)
    yz = (y / z)
    # 2. Apply radial camera distortion:
    rr = xz**2 + yz**2
    distortion = (1 + K1 * rr + K2 * rr * rr)
    xz *= distortion
    yz *= distortion
    # 3. Apply intrinsic matrix to get image coordinates:
    u = FX * xz + X0
    v = FY * yz + Y0
    return u, v

