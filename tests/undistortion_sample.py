import cv2
import numpy as np

cameraMatrix = np.array([[2628.39469, 0, 1439.72123],
                         [0, 2633.8381, 890.31892],
                         [0, 0, 1]])
distCoeffs = np.array([-0.344684, 0.124001, 0.000416, -0.000475])
imageSize = (2880, 1860)
alpha = 0

newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha)
print(newCameraMatrix)
print(roi)

# sample_image_path = '/home/zijiejiang/Documents/colmap_tmp_model/1604454430922404596.jpg'
# sample_image = cv2.imread(sample_image_path)
# dst = cv2.undistort(sample_image, cameraMatrix, distCoeffs, None, newCameraMatrix)

# dst = cv2.resize(dst, (720, 465), interpolation=cv2.INTER_LANCZOS4)

# cv2.imshow('Undistortion', dst)
# cv2.waitKey(0)
