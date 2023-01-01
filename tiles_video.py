# http://jepsonsblog.blogspot.tw/2012/11/rotation-in-3d-using-opencvs.html

import hashlib
import os
import sys
import re
import cv2
import numpy as np

TILE_SIZE = 256  # pixels
EARTH_CIRCUMFERENCE = 40075.016686 * 1000  # in meters, at the equator
EARTH_RADIUS_M = 6371000

def resolution(zoom, lat=0):
    m_per_pix = ((EARTH_CIRCUMFERENCE / TILE_SIZE) * np.cos(np.radians(lat))) / (2**zoom)
    return m_per_pix

def inv_mercator(mx, my, zoom):
    # GPS point for the top left corner of the tile
    factor = (1 / (2 * np.pi)) * 2 ** zoom
    lat = 2*np.arctan(np.exp(np.pi-(my/factor))) - (np.pi/2)
    lon = mx/factor - np.pi
    return (np.degrees(lat), np.degrees(lon))

def ned_translate(lat, lon, dN=0, dE=0):
    dLat = dN/EARTH_RADIUS_M
    dLon = dE/(EARTH_RADIUS_M*np.cos(np.radians(lat)))
    newlat = lat + np.degrees(dLat)
    newlon = lon + np.degrees(dLon)
    return (newlat, newlon)

def brown_rmax(k1, k2, k3):
    roots = np.roots([7*k3, 5*k2, 3*k1, 1])
    real = np.sqrt(np.abs(roots[np.isreal(roots)]))
    if len(real)>0:
        rmax = min(real[real>0])
    else:
        rmax = np.inf
    return rmax

class SimCam:
    def __init__(self, map_filename, cam_params, cam_att):
        self.map_filename = map_filename
        self.mx = int(re.search('swx(.*)-swy', map_filename).group(1))
        self.my = int(re.search('ney(.*)-', map_filename).group(1))
        self.zoom = int(re.search('z(.*).png', map_filename).group(1))
        self.tl_lat, self.tl_lon = inv_mercator(self.mx, self.my, self.zoom)
        self.m_per_pix = resolution(zoom=self.zoom, lat=self.tl_lat)
        self.ground_image = cv2.imread(self.map_filename)
        self.cam_params = cam_params
        self.cam_att = cam_att
        self.dst_pixels, self.invalid_pixels = self.create_distorted_pixels()
        
    def create_distorted_pixels(self):
        hash_filename = hashlib.sha256(str(self.cam_params).encode('utf-8')).hexdigest() + ".npy"
        print(f"{hash_filename = }")
        # Load the pixels if they exist or calculate them
        width, height = self.cam_params["width"], self.cam_params["height"]
        dst_pixels = np.ones((height, width, 3), np.float32)
        dst_pixels[:, :, :2] = np.mgrid[0:width, 0:height].T
        distortion = self.cam_params["distortion"]
        if np.sum(abs(distortion))>0:
            distortion_folder = r"SimCam/cam_params"
            os.makedirs(distortion_folder, exist_ok=True)
            src_filename = os.path.join(distortion_folder, hash_filename)
            # Check if the distortion pixels are saved
            if os.path.isfile(src_filename):
                # Load the params
                with open(src_filename, 'rb') as f:
                    dst_pixels = np.load(f)
                    invalid_pixels = np.load(f)
            else:
                # Calculate and save the params
                k1, k2, p1, p2, k3 = distortion
                # Convert to camera coordinates
                fx, fy = 2*self.cam_params["fx"], 2*self.cam_params["fy"]
                cx, cy = self.cam_params["cx"], self.cam_params["cy"]
                cameraMatrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1],
                ])
                # Find double prime, invert the projection matrix
                inv_M = np.linalg.inv(cameraMatrix)
                double_prime = dst_pixels.dot(inv_M.T)  # undistorted pixels
                rmax = brown_rmax(k1, k2, k3)
                x_pp_goal = double_prime[:,:,0].reshape(-1)
                y_pp_goal = double_prime[:,:,1].reshape(-1)
                # Initialize guess
                x_p = x_pp_goal.copy()
                y_p = y_pp_goal.copy()
                for i in range(1000):
                    r_s = x_p**2 + y_p**2
                    radial = 1 + k1*r_s + k2*r_s*r_s + k3*r_s*r_s*r_s
                    # Undistort guess to find the error
                    x_pp_hat = x_p*radial + 2*p1*x_p*y_p + p2*(r_s + 2*x_p*x_p)
                    y_pp_hat = y_p*radial + 2*p2*x_p*y_p + p1*(r_s + 2*y_p*y_p)
                    # Calculate the updates
                    dx_pp = x_pp_goal - x_pp_hat
                    dy_pp = y_pp_goal - y_pp_hat
                    dx_p = radial + 2*p1*y_p + 4*p2*x_p + 2*p2*y_p
                    dy_p = radial + 2*p1*x_p + 4*p2*y_p + 2*p2*x_p
                    alpha = 0.1
                    dx = alpha*(dx_pp*dx_p)
                    dy = alpha*(dy_pp*dy_p)
                    # Update the valid pixels only
                    valid = r_s < rmax*rmax
                    alpha = 1.0
                    x_p[valid] += alpha*dx[valid]
                    y_p[valid] += alpha*dy[valid]

                    errorX = np.abs(x_pp_goal[valid]-x_pp_hat[valid])
                    errorY = np.abs(y_pp_goal[valid]-y_pp_hat[valid])
                    print(f"{i}. X {np.sum(errorX):.5f}. {np.max(errorX):.5f}. Y {np.sum(errorY):.5f}. {np.max(errorY):.5f}.")

                    if np.max(errorX)<0.001 and np.max(errorY)<0.001:
                        break

                single_prime = np.ones((height, width, 3), np.float32)
                single_prime[:, :, 0] = x_p.reshape(height, width)
                single_prime[:, :, 1] = y_p.reshape(height, width)
                dst_pixels = single_prime.dot(cameraMatrix.T)
                invalid_pixels = ~valid.reshape(height, width)

                with open(src_filename, 'wb') as f:
                    np.save(f, dst_pixels)
                    np.save(f, invalid_pixels)
        else:
            # Do nothing and no pixels are invalid
            invalid_pixels = np.full((height, width,), False)
        return dst_pixels, invalid_pixels

    def lla_to_pixel(self, lat, lon, alt):
        """ Convert (lat, lon, alt) to (u,v,w) pixels"""
        dLat = np.radians(lat - self.tl_lat)
        dLon = np.radians(lon - self.tl_lon)
        dN = dLat*EARTH_RADIUS_M
        dE = dLon*(EARTH_RADIUS_M*np.cos(np.radians(self.tl_lat)))
        return [dE/self.m_per_pix, -dN/self.m_per_pix, alt/self.m_per_pix]

    def make_extrinsics(self, uav_pos, uav_att):
        h, w, c = self.ground_image.shape
        p = self.lla_to_pixel(*uav_pos)
        t = [w/2-p[0], h/2-p[1], p[2]]
        T = np.eye(4)
        T[:3,3] = t

        roll, pitch, yaw = uav_att
        croll, cpitch, cyaw = self.cam_att
        theta, phi, gamma = np.radians(-pitch-cpitch), np.radians(-roll-croll), np.radians(-yaw-cyaw)
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])
        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])
        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])

        # Projection 3D -> 2D matrix, camera intrinsics
        fx, fy = self.cam_params["fx"], self.cam_params["fy"]
        cx, cy = self.cam_params["cx"], self.cam_params["cy"]
        A2 = np.array([ [fx, 0, cx, 0],
                        [0, fy, cy, 0],
                        [0, 0, 1, 0]])
        
        # This is foward, map pixel to camera pixel
        # Original site uses this order
        # extrinsics = np.dot(A2, np.dot(T, np.dot(R, A1)))
        # But translate should be first bc it's aligned with the map pixels
        extrinsics = np.dot(A2, np.dot(R, np.dot(T, A1)))
        # print(extrinsics)
        return extrinsics

    def capture(self, uav_pos, uav_att):
        extrinsics = self.make_extrinsics(uav_pos, uav_att)
        inv_extrinsics = np.linalg.inv(extrinsics)
        # Start with pixels values of the distorted output image
        dst_pixels = self.dst_pixels.copy()
        # Calculate source image x, y
        src_pixels = dst_pixels.dot(inv_extrinsics.T)
        mapx = np.divide(src_pixels[:,:,0], src_pixels[:,:, 2]).astype(np.float32)
        mapy = np.divide(src_pixels[:,:,1], src_pixels[:,:, 2]).astype(np.float32)
        # Remove -z values, points not in the viewing frustum
        mapx[src_pixels[:,:, 2] < 0] = -1
        mapy[src_pixels[:,:, 2] < 0] = -1
        # Remove invalid pixels
        mapx[self.invalid_pixels] = -1
        mapy[self.invalid_pixels] = -1

        # Use cv2.remap
        out = cv2.remap(self.ground_image, mapx, mapy, cv2.INTER_LINEAR)

        # Capture the simulated perspective
        # Undistorted perfect image
        # ideal_lens = cv2.warpPerspective(
        #             src=self.ground_image,
        #             M=extrinsics,
        #             dsize=(width, height),
        #             flags=cv2.INTER_LINEAR,
        #             borderMode=cv2.BORDER_CONSTANT,
        #             borderValue=0,
        # )
        # cv2.imwrite("SimCam/out2.jpg", ideal_lens)

        # Apply undistort to compare with out2
        # cameraMatrix = np.array([
        #     [2*fx, 0, cx],
        #     [0, 2*fy, cy],
        #     [0, 0, 1],
        # ])
        # newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        #     cameraMatrix=cameraMatrix,
        #     distCoeffs=self.distortion,
        #     imageSize=(width, height),
        #     alpha=0,
        #     newImgSize=(width, height),
        # )
        # undistorted = cv2.undistort(
        #     src=out,
        #     cameraMatrix=cameraMatrix,
        #     distCoeffs=self.distortion,
        #     newCameraMatrix=newCameraMatrix,
        # )
        # cv2.imwrite(f"SimCam/out1.jpg", undistorted)


        # Show the simulated image on the gps image
        # out2 = cv2.warpPerspective(
        #             src=out,
        #             M=extrinsics,
        #             dsize=(w, h),
        #             flags=cv2.WARP_INVERSE_MAP,
        #         )
        return out

def main():
    # map_filename = "googlemaps-stitch/swx148701-swy196953-nex148704-ney196950-z19.png"
    map_filename = "googlemaps-stitch/swx297402-swy393907-nex297409-ney393901-z20.png"
    # map_filename = "googlemaps-stitch/swx594804-swy787815-nex594819-ney787803-z21.png"
    # map_filename = "googlemaps-stitch/swx1189609-swy1575631-nex1189639-ney1575606-z22.png"
    # map_filename = "googlemaps-stitch/swx2379218-swy3151263-nex2379279-ney3151213-z23.png"

    # map_filename = "googlemaps-stitch/swx297400-swy393909-nex297412-ney393899-z20.png"
    # map_filename = "googlemaps-stitch/swx594800-swy787819-nex594824-ney787799-z21.png"

    # lat, lon, alt in meters
    # above home plate
    uav_pos = [40.80164174483064, -77.89348745160423, 10]
    # roll, pitch, yaw, degrees
    uav_att = [0, 0, 0]
    # camera attitude offsets from straight down
    cam_att = [0, 90-12, 0]

    # calibrate gopro hero 3 1080p wide
    distortion = np.array([
        -2.71969672e-01,    # k1
        1.12979477e-01,     # k2
        6.30516626e-06,     # p1
        2.62608080e-04,     # p2
        -2.82170416e-02,    # k3
    ])
    cam_params = {
        "width": 1920,
        "height": 1080,
        "fx": 860.42491325/2,
        "fy": 878.34592761/2,
        "cx": 949.41289961,
        "cy": 498.99907074,
        "distortion": distortion,
    }

    sim = SimCam(
            map_filename=map_filename,
            cam_params=cam_params,
            cam_att=cam_att,
        )

    out = sim.capture(uav_pos, uav_att)
    cv2.imwrite("SimCam/out0.jpg", out)
    exit()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30.0
    video_writer = cv2.VideoWriter("SimCam/out.avi", fourcc, fps, image_size)
    max_pitch = 120
    max_roll = 60
    steps = 60
    for t in range(2*steps):
        r = np.radians(t*90/steps)
        y = t*360/steps
        s, c = np.sin(r), np.cos(0.7*r)
        uav_att = [max_roll*c, max_pitch*s, y]
        out = sim.capture(uav_pos, uav_att)
        # cv2.imwrite(f"SimCam/out_{t:02d}.jpg", out)
        video_writer.write(out)
    video_writer.release()

    # move aligned data to utils
    # Interpolate between rows for given time
    # Use last for any categorical modes
    # Load log
    # Find start and end of flight +- 10 sec
    # Render at 1 fps and compare to gopro

if __name__ == "__main__":
    main()
