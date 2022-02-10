import numpy as np
import open3d as o3d
import cv2
import collections

from evaluate_ibl import load_pose_pickle


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


pcd_path = '/home/zijiejiang/shin_simulation/map/pointcloud_map.pcd'
pcd = o3d.io.read_point_cloud(pcd_path)
o3d.visualization.draw_geometries([pcd])

w = 1280
h = 384
fx = 1039.13693
fy = 1038.90465
cx = 720.19014 - 80
cy = 553.13684 - 348
k1 = -0.117387
k2 = 0.087465
ts = 1602639072332893465

pose_pickle_path = '/home/zijiejiang/Documents/dataset/shinjuku_d_1014_q_1012/processed/database/pose.pickle'
pose_dict = load_pose_pickle(pose_pickle_path)

T_map_cam = pose_dict['{}.jpg'.format(ts)]
T_cam_map = np.linalg.inv(T_map_cam)
pcd_points = np.asarray(pcd.points)
pcd_points = np.hstack([pcd_points, np.ones((len(pcd_points), 1))])
cam_points = T_cam_map @ np.transpose(pcd_points)
cam_points = np.transpose(cam_points)[:, :3]

xyz_view = get_camera_view_pointcloud(cam_points, h, w, k1, k2, fx, fy, cx, cy)
occluded_inds = get_occluded_points(xyz_view, 0.002, 0.08)
occluded_inds = set(occluded_inds)
visible_indices = [
    i for i in range(xyz_view.shape[0]) if i not in occluded_inds
]
visible_xyz = xyz_view[visible_indices, :]
visible_xyz = filter_out_ot_frame_points(visible_xyz, h, w, k1, k2, fx, fy, cx, cy)
inds_to_keep = sample_uniform(visible_xyz, 1e-3)
visible_xyz = visible_xyz[inds_to_keep]

z_image = render_z(visible_xyz, h, w, k1, k2, fx, fy, cx, cy)
depth_image = (z_image * 256).astype(np.uint16)

image_jet = cv2.applyColorMap(
        np.uint8(z_image / np.amax(z_image) * 255),
        cv2.COLORMAP_JET)
# cv2.imshow('Depth', image_jet)
# cv2.waitKey(0)

pcd.points = o3d.utility.Vector3dVector(visible_xyz)
o3d.visualization.draw_geometries([pcd])
