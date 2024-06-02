import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

height = 1080
width = 1920
dst_pixels = np.ones((height, width, 3), np.float32)
# print(f"{dst_pixels.shape = }")
dst_pixels[:, :, :2] = np.mgrid[0:width, 0:height].T
# print(f"{dst_pixels.shape = }")
# print(f"{dst_pixels = }")
# print(dst_pixels[:,:,0])  # all the x coordinates
# print(dst_pixels[:,:,1])  # all the y coordinates
# print(dst_pixels[:,:,2])  # all the 1

extrinsics = np.array(
[[ 4.30212457e+02,  0.00000000e+00, -6.60211129e+05],
 [ 0.00000000e+00,  4.39172964e+02, -3.84638574e+05],
 [ 0.00000000e+00,  0.00000000e+00,  8.94892198e+01],]
)
inv_extrinsics = np.linalg.inv(extrinsics)

corners = np.array([
    [[0, 0, 1], [width, 0, 1]],
    [[0, height, 1], [width, height, 1]],    
])
# print(f"{corners.shape = }")
# print(f"{corners = }")

"""
0 [1.71486185e+01 9.78693080e+00 1.11745303e-02]
1 [2.16115297e+01 9.78693080e+00 1.11745303e-02]
2 [2.16115297e+01 1.22460986e+01 1.11745303e-02]
3 [1.71486185e+01 1.22460986e+01 1.11745303e-02]
"""

src_pixels = corners.dot(inv_extrinsics.T)
src_pixels[:,:,0] = np.divide(src_pixels[:,:,0], src_pixels[:,:, 2])
# print(src_pixels)

fx, fy = 860.42491325, 878.34592761
cx, cy = 949.41289961, 498.99907074
# fx, fy = width/2, height/2
# cx, cy = width/2, height/2
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
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
    cameraMatrix=cameraMatrix,
    distCoeffs=distortion,
    imageSize=(width, height),
    alpha=0,
    newImgSize=(width, height),
)

def single_to_double(x_p, y_p, distortion):
    k1, k2, p1, p2, k3 = distortion
    r_s = x_p**2 + y_p**2
    radial = 1 + k1*r_s + k2*r_s*r_s + k3*r_s*r_s*r_s
    x_pp = x_p*radial + 2*p1*x_p*y_p + p2*(r_s + 2*x_p*x_p)
    y_pp = y_p*radial + 2*p2*x_p*y_p + p1*(r_s + 2*y_p*y_p)
    return [x_pp, y_pp]

def double_to_single(double, distortion):
    k1, k2, p1, p2, k3 = distortion

    # http://wscg.zcu.cz/WSCG2018/Poster/P83-full.PDF
    # x_pp = double[0,:,0].reshape(-1)
    # y_pp = double[0,:,1].reshape(-1)
    # r_s = x_pp**2 + y_pp**2
    # d1 = k1*r_s + k2*r_s*r_s + k3*r_s*r_s*r_s
    # d2 = 1 / (4*k1*r_s + 6*k2*r_s*r_s + 8*p1*y_pp + 8*p2*x_pp + 1)
    # x_p = x_pp - d2 * ( x_pp*d1 + 2*p1*x_pp*y_pp + p2*(r_s + 2*x_pp*x_pp) )
    # y_p = y_pp - d2 * ( y_pp*d1 + 2*p2*x_pp*y_pp + p1*(r_s + 2*y_pp*y_pp) )

    x_pp_goal = double[0,:,0].reshape(-1)
    y_pp_goal = double[0,:,1].reshape(-1)
    x_p = x_pp_goal
    y_p = y_pp_goal
    for i in range(2000):
        r_s = x_p**2 + y_p**2
        R = 1 + k1*r_s + k2*r_s*r_s + k3*r_s*r_s*r_s
        # dx_pp = x_pp_goal - x_pP_hat
        # dy_pp = y_pp_goal - y_pP_hat
        # dx_p = R + 2*p1*y_p + 4*p2*x_p + 2*p2*y_p
        # dy_p = R + 2*p1*x_p + 4*p2*y_p + 2*p2*x_p
        # alpha = 0.1
        # x_p += alpha*(dx_p)
        # y_p += alpha*(dy_p)

        dtx = 2*p1*x_p*y_p + p2*(r_s+2*x_p*x_p)
        dty = 2*p2*x_p*y_p + p1*(r_s + 2*y_p*y_p)
        dR = 1*r_s + 2*k2*r_s + 3*k3*r_s*r_s  # dR/dr_s
        J1x = (6*p2*x_p + 2*p1*y_p) + (R + 2*x_p*x_p*dR)
        J1y = (2*p1*x_p + 2*p2*y_p) + (2*x_p*y_p*dR)
        J2x = J1y
        J2y = (2*p2*x_p + 6*p1*y_p) + (R + 2*y_p*y_p*dR)
        # Update
        den = J1x*J2y + J2x*J1y
        resx = x_pp_goal - (R*x_p + dtx)
        resy = y_pp_goal - (R*y_p + dty)
        dx_p = +(J2y/den)*resx - (J1y/den)*resy
        dy_p = -(J2x/den)*resx + (J1x/den)*resy

        alpha = 1.0
        x_p += alpha*(dx_p)
        y_p += alpha*(dy_p)

        x_pp_hat, y_pp_hat = single_to_double(x_p, y_p, distortion)
        # print(i)
        if i%100==0:
            print(f"{i}. errX = {np.sum(np.abs(x_pp_goal-x_pp_hat))}. errY = {np.sum(np.abs(y_pp_goal-y_pp_hat))}")
    # print(f"{x_p[:2] = } {y_p[:2] = }")

    exit()
    


    x_re, y_re = single_to_double(x_p, y_p, distortion)
    print(f"{x_p[:2] = } {y_p[:2] = }")
    print(f"{x_re[:2] = } {y_re[:2] = }")
    print(f"{x_pp[:2] = } {y_pp[:2] = }")
    exit()

    height, width, c = double.shape
    single = np.ones((height, width, 3), np.float32)
    single[:, :, 0] = x_p.reshape(height, width)
    single[:, :, 1] = y_p.reshape(height, width)
    return single

def distortion_map(cameraMatrix, distortion, imageSize):
    width, height = imageSize
    inv_M = np.linalg.inv(cameraMatrix)
    # Given a set of image distorted pixels
    dst_pixels = np.ones((height, width, 3), np.float32)
    dst_pixels[:, :, :2] = np.mgrid[0:width, 0:height].T


    gap = 60
    row = np.repeat(np.arange(0, width, gap), height//gap)
    column = np.tile(np.arange(0, height, gap), width//gap)
    dst_pixels = np.vstack((column, row)) #.reshape((height, width, 3))
    # np.ones(height*width)

    print(dst_pixels)
    exit()
    # Find double prime, invert the projection matrix
    double_prime = dst_pixels.dot(inv_M.T)
    # Use double prime to find single prime
    single_prime = double_to_single(double_prime, distortion).reshape(height, width, 3)
    # Convert single prime back to undistorted pixels
    real_pixels = single_prime.dot(cameraMatrix.T)
    return real_pixels[:,:,0], real_pixels[:,:,1]

mapx, mapy = distortion_map(
    cameraMatrix=cameraMatrix,
    distortion=distortion,
    imageSize=(width, height),
)
# print(mapx)
exit()


dst_pixels = np.ones((height, width, 3), np.float32)
dst_pixels[:, :, :2] = np.mgrid[0:width, 0:height].T

# dst_pixels = np.mgrid[0:width, 0:height].T
# dst_pixels = dst_pixels.reshape(height*width, 1, 2).astype(np.float32)
# h_corners = cv2.undistortPoints(dst_pixels, cameraMatrix, distortion)
# print(h_corners)
# h_corners = np.c_[h_corners.squeeze(), np.ones(len(h_corners))]
# print(h_corners)
# print(h_corners.shape)

# inv_M = np.linalg.inv(cameraMatrix)
# double_prime = dst_pixels.dot(inv_M.T)
# objectPoints = double_prime.reshape(height*width, 3)

# row = np.repeat(np.linspace(-1, 1, width), height)
# column = np.tile(np.linspace(-1, 1, height), width)
# objectPoints = np.vstack((column, row, np.ones(height*width))).


# projs, _ = cv2.projectPoints(
#     objectPoints=objectPoints,
#     rvec=(0,0,0),
#     tvec=(0,0,0),
#     cameraMatrix=cameraMatrix,
#     # distCoeffs=None,
#     distCoeffs=distortion,
# )
# print(projs)
# projs = projs.reshape(height, width, 2)
# mapx = projs[:,:, 0]
# mapy = projs[:,:, 1]

mapx2, mapy2 = cv2.initUndistortRectifyMap(
    cameraMatrix=cameraMatrix,
    distCoeffs=distortion,
    R=None,
    newCameraMatrix=cameraMatrix,
    size=(width, height),
    m1type=5,
)
# print(mapx2)
# exit()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(4,4))
# fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
ax1.scatter(mapx.reshape(-1), mapy.reshape(-1), s=1, color='red')
ax1.axis('equal')
ax2.scatter(mapx2.reshape(-1), mapy2.reshape(-1), s=1, color='red')
ax2.axis('equal')
# scale = 500
# ax1.set_xlim([-scale, width+scale])
# ax1.set_ylim([-scale, height+scale])
# ax2.scatter(mapx2.reshape(-1), mapy2.reshape(-1), s=1, color='red')
fig.tight_layout()
plt.show()
