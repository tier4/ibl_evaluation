import open3d as o3d
import numpy as np
import cv2

ts = 1602484279208413917
rgb_path = '/home/zijiejiang/mount/shinjuku_d_1014_q_1012/processed_lidar/database/image/{}.jpg'.format(ts)
depth_path = '/home/zijiejiang/mount/shinjuku_d_1014_q_1012/processed_lidar/database/depth_complete/{}.png'.format(ts)

color_raw = cv2.imread(rgb_path)
depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

color_raw = o3d.geometry.Image(color_raw)
depth_raw = o3d.geometry.Image(depth_raw)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=256.0, depth_trunc=100, convert_rgb_to_intensity=False)

fx = 1039.13693
fy = 1038.90465
cx = 720.19014 - 80
cy = 553.13684 - 348
intrinsic = o3d.camera.PinholeCameraIntrinsic(int(cx*2), int(cy*2), fx, fy, cx, cy)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
o3d.visualization.draw_geometries([pcd])
