import cv2
import numpy as np
import matplotlib.pyplot as plt

def single_to_double(x_p, y_p, distortion):
    k1, k2, p1, p2, k3 = distortion
    r_s = x_p**2 + y_p**2
    radial = 1 + k1*r_s + k2*r_s*r_s + k3*r_s*r_s*r_s
    x_pp = x_p*radial + 2*p1*x_p*y_p + p2*(r_s + 2*x_p*x_p)
    y_pp = y_p*radial + 2*p2*x_p*y_p + p1*(r_s + 2*y_p*y_p)
    return [x_pp, y_pp]

def brown_rmax(k1, k2, k3):
    roots = np.roots([7*k3, 5*k2, 3*k1, 1])
    real = np.sqrt(np.abs(roots[np.isreal(roots)]))
    if len(real)>0:
        rmax = min(real[real>0])
    else:
        rmax = np.inf
    return rmax

height = 1080
width = 1920
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
k1, k2, p1, p2, k3 = distortion
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
    cameraMatrix=cameraMatrix,
    distCoeffs=distortion,
    imageSize=(width, height),
    alpha=0,
    newImgSize=(width, height),
)

# Sample a bunch of pixels in the undistored image
gap = 120
row = np.repeat(np.arange(0, height, gap), width//gap)
column = np.tile(np.arange(0, width, gap), height//gap)
dst_pixels = np.vstack((column, row, np.ones(height//gap*width//gap)))
dst_pixels = dst_pixels.T.reshape((height//gap, width//gap, 3))

# Find double prime, invert the projection matrix
inv_M = np.linalg.inv(cameraMatrix)
double_prime = dst_pixels.dot(inv_M.T)

rmax = brown_rmax(k1, k2, k3)
print(f"{rmax = }")

# Make a figure
scale = 0.25
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(scale*16,scale*18))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 15.0
video_writer = cv2.VideoWriter("SimCam/distort3_try5/out.avi", fourcc, fps, (800,900))

# Use double prime (undistorted) to find single prime (distorted)
x_pp_goal = double_prime[:,:,0].reshape(-1)
y_pp_goal = double_prime[:,:,1].reshape(-1)
# Initialize guess
x_p = x_pp_goal.copy()
y_p = y_pp_goal.copy()
still_valid = np.full(y_p.shape, True)
for i in range(1000):

    # Undistort and check the error
    x_pp_hat, y_pp_hat = single_to_double(x_p, y_p, distortion)

    # r_s = x_p**2 + y_p**2
    # d1 = k1*r_s + k2*r_s*r_s + k3*r_s*r_s*r_s
    # d2 = 1 / (4*k1*r_s + 6*k2*r_s*r_s + 8*p1*y_p + 8*p2*x_p + 1)
    # dx = d2 * ( x_p*d1 + 2*p1*x_p*y_p + p2*(r_s + 2*x_p*x_p) )
    # dy = d2 * ( y_p*d1 + 2*p2*x_p*y_p + p1*(r_s + 2*y_p*y_p) )

    # r_s = x_p**2 + y_p**2
    # R = 1 + k1*r_s + k2*r_s*r_s + k3*r_s*r_s*r_s
    # dtx = 2*p1*x_p*y_p + p2*(r_s+2*x_p*x_p)
    # dty = 2*p2*x_p*y_p + p1*(r_s + 2*y_p*y_p)
    # dR = 1*r_s + 2*k2*r_s + 3*k3*r_s*r_s  # dR/dr_s
    # J1x = (6*p2*x_p + 2*p1*y_p) + (R + 2*x_p*x_p*dR)
    # J1y = (2*p1*x_p + 2*p2*y_p) + (2*x_p*y_p*dR)
    # J2x = J1y
    # J2y = (2*p2*x_p + 6*p1*y_p) + (R + 2*y_p*y_p*dR)
    # den = J1x*J2y + J2x*J1y
    # resx = x_pp_goal - (R*x_p + dtx)
    # resy = y_pp_goal - (R*y_p + dty)
    # dx_p = +(J2y/den)*resx - (J1y/den)*resy
    # dy_p = -(J2x/den)*resx + (J1x/den)*resy
    # dx = -dx_p
    # dy = -dy_p

    r_s = x_p**2 + y_p**2
    R = 1 + k1*r_s + k2*r_s*r_s + k3*r_s*r_s*r_s
    dx_pp = x_pp_goal - x_pp_hat
    dy_pp = y_pp_goal - y_pp_hat
    dx_p = R + 2*p1*y_p + 4*p2*x_p + 2*p2*y_p
    dy_p = R + 2*p1*x_p + 4*p2*y_p + 2*p2*x_p
    alpha = 0.1
    dx = alpha*(dx_pp*dx_p)
    dy = alpha*(dy_pp*dy_p)
    # print(dx_pp[:4])
    # print(dx_p[:4])

    # Update the valid pixels only
    valid = r_s < rmax*rmax
    alpha = 1.0
    x_p[valid] += alpha*dx[valid]
    y_p[valid] += alpha*dy[valid]

    # Calculate the distorted pixels
    h, w, c = dst_pixels.shape
    single_prime = np.ones((h, w, 3), np.float32)
    single_prime[:, :, 0] = x_p.reshape(h, w)
    single_prime[:, :, 1] = y_p.reshape(h, w)
    real_pixels = single_prime.dot(cameraMatrix.T)
    # x_distort, y_distort = real_pixels[:,:,0], real_pixels[:,:,1]
    x_distort, y_distort = single_prime[:,:,0], single_prime[:,:,1]

    # Calculate the undisorted pixels
    single_prime = np.ones((h, w, 3), np.float32)
    single_prime[:, :, 0] = x_pp_hat.reshape(h, w)
    single_prime[:, :, 1] = y_pp_hat.reshape(h, w)
    real_pixels = single_prime.dot(cameraMatrix.T)
    # x_undistort, y_undistort = real_pixels[:,:,0], real_pixels[:,:,1]
    # x_true, y_true = dst_pixels[:,:,0], dst_pixels[:,:,1]
    x_undistort, y_undistort = single_prime[:,:,0], single_prime[:,:,1]
    x_true, y_true = double_prime[:,:,0], double_prime[:,:,1]

    # Draw frame
    if i%5==0:
        errorX = np.abs(x_pp_goal[valid]-x_pp_hat[valid])
        errorY = np.abs(y_pp_goal[valid]-y_pp_hat[valid])
        print(f"{i}. X {np.sum(errorX):.5f}. {np.max(errorX):.5f}. Y {np.sum(errorY):.5f}. {np.max(errorY):.5f}.")

        if np.max(errorX)<0.001 and np.max(errorY)<0.001:
            break

        ax1.clear()
        if not np.isinf(rmax):
            circle1 = plt.Circle((0, 0), rmax, color='g', fill=False)
            ax1.add_patch(circle1)
        ax1.grid()
        ax1.scatter(x_distort.reshape(-1)[valid], y_distort.reshape(-1)[valid], s=1, color='red')
        ax1.set_title('Distorted pixels')
        ax1.axis('equal')
        bounds = 2000
        # ax1.set_xlim([-bounds, width+bounds])
        # ax1.set_ylim([-bounds, height+bounds])

        ax2.clear()
        if not np.isinf(rmax):
            circle2 = plt.Circle((0, 0), rmax, color='g', fill=False)
            ax2.add_patch(circle2)
        ax2.grid()
        ax2.scatter(x_true.reshape(-1), y_true.reshape(-1), s=1, color='green')
        ax2.scatter(x_undistort.reshape(-1)[valid], y_undistort.reshape(-1)[valid], s=0.5, color='blue')
        ax2.set_title('Undistort error')
        ax2.axis('equal')
        # ax2.set_xlim([-bounds, width+bounds])
        # ax2.set_ylim([-bounds, height+bounds])
        fig.tight_layout()
        frame_filename = f'SimCam/distort3_try5/{i:04d}.png'
        fig.savefig(frame_filename, dpi=200)

        frame = cv2.imread(frame_filename)
        video_writer.write(frame)

video_writer.release()

# plt.show()

exit()
