import cv2
import numpy as np


fx, fy = 860.42491325, 878.34592761
cx, cy = 949.41289961, 498.99907074
cameraMatrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1],
])
distortion = np.array([
    -2.71969672e-01,    # k1
    1.12979477e-01,     # k2
    6.30516626e-06,     # p1
    2.62608080e-04,     # p2
    -2.82170416e-02,    # k3
])


filename = r"cam_calibration\images\gopro_hero_3_1080_w\gopro_1_445.jpg"
original_img = cv2.imread(filename)
height, width, c = original_img.shape



gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 60, 0.001)
corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
print(corners.shape)
print(corners)

h_corners = cv2.undistortPoints(corners, cameraMatrix, distortion)
print(h_corners.shape)
h_corners = np.c_[h_corners.squeeze(), np.ones(len(h_corners))]
print(h_corners.shape)
print(h_corners)
exit()

img_pts, _ = cv2.projectPoints(h_corners, (0, 0, 0), (0, 0, 0), cameraMatrix, None)
img_pts_dest, _ = cv2.projectPoints(h_corners, (0, 0, 0), (0, 0, 0), cameraMatrix, distortion)
for c in corners:
    cx, cy = c[0]
    cv2.circle(original_img, (int(cx), int(cy)), 10, (0, 255, 0), 2)
for c in img_pts.squeeze().astype(np.float32):
    cx, cy = c
    cv2.circle(original_img, (int(cx), int(cy)), 5, (0, 0, 255), 2)
for c in img_pts_dest.squeeze().astype(np.float32):
    cx, cy = c
    cv2.circle(original_img, (int(cx), int(cy)), 5, (255, 255, 0), 2)

cv2.imshow('Projection', cv2.resize(original_img, dsize=(width//2, height//2)))
cv2.waitKey()
cv2.destroyAllWindows()

exit()

newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(
    cameraMatrix=K,
    distCoeffs=distortion,
    imageSize=(width, height),
    alpha=0,
    newImgSize = (width, height),
)
print(K)
print(newcameramatrix)
print(roi)

undistorted = cv2.undistort(
    src=original_img,
    cameraMatrix=K,
    distCoeffs=distortion,
    newCameraMatrix=newcameramatrix,
)
cv2.imwrite(f"SimCam/img00.jpg", undistorted)

reundistorted = cv2.undistort(
    src=undistorted,
    cameraMatrix=K,
    distCoeffs=-distortion,
    newCameraMatrix=newcameramatrix,
)
cv2.imwrite(f"SimCam/img01.jpg", reundistorted)

# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite(f"SimCam/undistort03.jpg", dst)

# mapx, mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
# dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
