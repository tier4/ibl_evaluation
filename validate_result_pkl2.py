import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from evaluate_ibl import shift_origin

np.set_printoptions(suppress=True)


def load_pkl(path):
    with open(path, 'rb') as f:
        result_dict = pickle.load(f)

    return result_dict


def load_pose_pickle(path):
    pose_dict = {}

    df = pd.read_pickle(path)
    img_path_list = list(df.loc[:, 'image_name'])
    img_path_list = [path.split('/')[-1] for path in img_path_list]
    img_pose_list = np.array(list(df.loc[:, 'image_pose'].values))

    for idx, img_pose in enumerate(img_pose_list):
        pose_dict[img_path_list[idx]] = img_pose

    return pose_dict


def convert_to_dict(kpts, map_points, inliers):
    info_dict = {}
    for kpt, map_point, is_inlier in zip(kpts, map_points, inliers):
        if not kpt.tobytes() in info_dict:
            info_dict[kpt.tobytes()] = [[kpt, map_point, is_inlier]]
        else:
            info_dict[kpt.tobytes()].append([kpt, map_point, is_inlier])

    return info_dict


image_result_pkl_path = '/home/zijiejiang/mount/shinjuku_d_1014_q_1012/processed_sfm/output/result_log.pkl'
gt_result_path = '/home/zijiejiang/mount/shinjuku_d_1014_q_1012/processed_sfm/query/pose.pickle'
processed_dir = '/home/zijiejiang/mount/shinjuku_d_1014_q_1012/processed_sfm'
origin = np.array([81819.65625, 50382.8984375, 40.998046875])

image_result_dict = load_pkl(image_result_pkl_path)
gt_result_dict = load_pose_pickle(gt_result_path)
shift_origin(gt_result_dict, origin)

# ts = 1602484279208413917
# ts = 1602484292909657750
ts = 1602484223515190443
qname = 'query/image/{}.jpg'.format(ts)

print("### Image-Only Statics ###")
print('Number of Inliers: {}'.format(image_result_dict['loc'][qname]['PnP_ret']['num_inliers']))
print('Total matchings: {}'.format(len(image_result_dict['loc'][qname]['PnP_ret']['inliers'])))
# print(len(image_result_dict['loc'][qname]['keypoints_query']))
# print(len(image_result_dict['loc'][qname]['points3D_xyz']))

########################
# Same 2D point
image_kpts = np.round(image_result_dict['loc'][qname]['keypoints_query'], 1)
image_3ds = np.round(image_result_dict['loc'][qname]['points3D_xyz'], 1)
image_inliers = image_result_dict['loc'][qname]['PnP_ret']['inliers']
image_info_dict = convert_to_dict(image_kpts, image_3ds, image_inliers)

print('The number of matched 2D points (image): {}'.format(len(image_info_dict)))

image_data = plt.imread(os.path.join(processed_dir, qname))
h, w, _ = image_data.shape

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
cmap = cm.autumn

ax.set_title('Use SfM Depth')
for _, val_list in image_info_dict.items():
    kpt = val_list[0][0]
    total = 0
    inlier_num = 0
    for val in val_list:
        total += 1
        if val[2]:
            inlier_num += 1

    ax.scatter(x=kpt[0],
                  y=kpt[1],
                  color=cmap(inlier_num / total), s=4)


ax.imshow(image_data)
ax.axis('off')
plt.tight_layout()
plt.show()

# count_in = 0
# count_out = 0
# for kpt in image_kpts:
#     if len(np.where(np.prod(lidar_kpts == kpt, axis = -1))[0]) > 0:
#         count_in += 1
#         # print(kpt)
#         print(np.where(np.prod(lidar_kpts == kpt, axis = -1)))
#     else:
#         count_out += 1

# print('')
# print('### In Number : {}'.format(count_in))
# print('### Out Number : {}'.format(count_out))

# with open('2D-3D_lidar.txt', 'w') as f:
#     for _, val_list in lidar_info_dict.items():
#         for val in val_list:
#             kpt, map_point, is_inlier = val
#             f.write('2D: {} ---> 3D: {}, {}\n'.format(kpt, np.round(map_point, 1), is_inlier))
#         f.write('\n')
