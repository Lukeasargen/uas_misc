# http://jepsonsblog.blogspot.tw/2012/11/rotation-in-3d-using-opencvs.html


import cv2
import numpy as np

TILE_SIZE = 256  # pixels
EARTH_CIRCUMFERENCE = 40075.016686 * 1000  # in meters, at the equator
EARTH_RADIUS_M = 6371000
def cosd(x):
    return np.cos(np.radians(x))

def sind(x):
    return np.sin(np.radians(x))

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

class SimCam:
    def __init__(self, mx, my, zoom, map_filename, image_size,
            cam_att, focal_length, optical_center, distortion):
        self.mx = mx
        self.my = my
        self.zoom = zoom
        self.map_filename = map_filename
        tl_lat, tl_lon = inv_mercator(mx, my, zoom)
        self.tl_lat = tl_lat
        self.tl_lon = tl_lon
        m_per_pix = resolution(zoom=zoom, lat=tl_lat)
        self.m_per_pix = m_per_pix
        self.ground_image = cv2.imread(self.map_filename)
        # Replace the top corner with black to remap a "blank" pixel
        self.ground_image[0][0] = np.zeros((1,3))
        self.image_size = image_size
        self.cam_att = cam_att
        self.focal_length = focal_length
        self.optical_center = optical_center
        self.distortion = distortion

    def lla_to_pixel(self, lat, lon, alt):
        """ Convert (lat, lon, alt) to (u,v,w) pixels"""
        dLat = np.radians(lat - self.tl_lat)
        dLon = np.radians(lon - self.tl_lon)
        dN = dLat*EARTH_RADIUS_M
        dE = dLon*(EARTH_RADIUS_M*np.cos(np.radians(self.tl_lat)))
        return [dE/self.m_per_pix, -dN/self.m_per_pix, alt/self.m_per_pix]

    def capture(self, uav_pos, uav_att):
        h, w, c = self.ground_image.shape
        width, height = self.image_size
        fx, fy = self.focal_length
        cx, cy = self.optical_center

        # Not used, but useful formulas for total FOV angle
        # fov_x = np.degrees(2*np.arctan(0.5*width/fx))
        # fov_y = np.degrees(2*np.arctan(0.5*height/fy))
        # print(f"{fov_x = }. {fov_y = }")

        # camera extrinsics
        p = self.lla_to_pixel(*uav_pos)
        # print("uav pos pixel :", p)
        t = [w/2-p[0], h/2-p[1], p[2]]
        # print(f"{t = }")
        T = np.eye(4)
        T[:3,3] = t

        roll, pitch, yaw = uav_att
        croll, cpitch, cyaw = self.cam_att
        theta, phi, gamma = np.radians(-pitch-croll), np.radians(-roll-cpitch), np.radians(-yaw-cyaw)
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
        A2 = np.array([ [fx, 0, cx, 0],
                        [0, fy, cy, 0],
                        [0, 0, 1, 0]])
        
        # This is foward, map pixel to camera pixel
        # Original site uses this order
        # extrinsics = np.dot(A2, np.dot(T, np.dot(R, A1)))
        # But translate should be first bc it's aligned with the map pixels
        extrinsics = np.dot(A2, np.dot(R, np.dot(T, A1)))
        inv_extrinsics = np.linalg.inv(extrinsics)
        # print(extrinsics)

        # Start with pixels values of the output image
        dst_pixels = np.ones((height, width, 3), np.float32)
        dst_pixels[:, :, :2] = np.mgrid[0:width, 0:height].T

        # TODO: undistorted image -> distorted pixels from full FOV

        # Calculate source image x, y
        src_pixels = dst_pixels.dot(inv_extrinsics.T)
        mapx = np.divide(src_pixels[:,:,0], src_pixels[:,:, 2]).astype(np.float32)
        mapy = np.divide(src_pixels[:,:,1], src_pixels[:,:, 2]).astype(np.float32)
        # Remove -z values, points not in the viewing frustum
        mapx[src_pixels[:,:, 2] < 0] = 0
        mapy[src_pixels[:,:, 2] < 0] = 0
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
    # mx, my, zoom = 148701, 196950, 19  # home plate (610, 360)
    # map_filename = "googlemaps-stitch/swx297402-swy393907-nex297409-ney393901-z20.png"
    # mx, my, zoom = 297402, 393901, 20
    # map_filename = "googlemaps-stitch/swx594804-swy787815-nex594819-ney787803-z21.png"
    # mx, my, zoom = 594804, 787803, 21  # home plate (2441, 676)
    # map_filename = "googlemaps-stitch/swx1189609-swy1575631-nex1189639-ney1575606-z22.png"
    # mx, my, zoom = 1189609, 1575606, 22
    # map_filename = "googlemaps-stitch/swx2379218-swy3151263-nex2379279-ney3151213-z23.png"
    # mx, my, zoom = 2379218, 3151213, 23

    map_filename = "googlemaps-stitch/swx297400-swy393909-nex297412-ney393899-z20.png"
    mx, my, zoom = 297400, 393899, 20
    # map_filename = "googlemaps-stitch/swx594800-swy787819-nex594824-ney787799-z21.png"
    # mx, my, zoom = 594800, 787799, 21

    # lat, lon, alt in meters
    # above home plate
    uav_pos = [40.80164174483064, -77.89348745160423, 10]
    # roll, pitch, yaw, degrees
    uav_att = [0, 0, 0]
    # camera attitude offsets from straight down
    cam_att = [0, 0, 0]

    # this is approximately a gopro hero 3
    # width, height, in pixels, cam resolution
    image_size = 1920, 1080
    # focal_length = 472, 476
    # optical_center = 960, 540
    focal_length = 860.42491325/2, 878.34592761/2
    optical_center = 949.41289961, 498.99907074
    distortion = np.array([
        -2.71969672e-01,    # k1
        1.12979477e-01,     # k2
        6.30516626e-06,     # p1
        2.62608080e-04,     # p2
        -2.82170416e-02,    # k3
    ])

    sim = SimCam(
            mx=mx, 
            my=my, 
            zoom=zoom, 
            map_filename=map_filename,
            cam_att=cam_att,
            image_size=image_size,
            focal_length=focal_length,
            optical_center=optical_center,
            distortion=distortion,
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
