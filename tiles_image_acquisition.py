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

class Ground:
    def __init__(self, mx, my, zoom, file) -> None:
        self.mx = mx
        self.my = my
        self.zoom = zoom
        self.map_filename = file
        tl_lat, tl_lon = inv_mercator(mx, my, zoom)
        self.tl_lat = tl_lat
        self.tl_lon = tl_lon
        m_per_pix = resolution(zoom=zoom, lat=tl_lat)
        self.m_per_pix = m_per_pix

    def lla_to_pixel(self, lat, lon, alt):
        """ Convert (lat, lon, alt) to (u,v,w) pixels"""
        dLat = np.radians(lat - self.tl_lat)
        dLon = np.radians(lon - self.tl_lon)
        dN = dLat*EARTH_RADIUS_M
        dE = dLon*(EARTH_RADIUS_M*np.cos(np.radians(self.tl_lat)))
        return [dE/self.m_per_pix, -dN/self.m_per_pix, alt/self.m_per_pix]

def main():
    map_filename = "googlemaps-stitch/swx148701-swy196953-nex148704-ney196950-z19.png"
    mx, my, zoom = 148701, 196950, 19  # home plate (610, 360)
    map_filename = "googlemaps-stitch/swx297402-swy393907-nex297409-ney393901-z20.png"
    mx, my, zoom = 297402, 393901, 20
    map_filename = "googlemaps-stitch/swx594804-swy787815-nex594819-ney787803-z21.png"
    mx, my, zoom = 594804, 787803, 21  # home plate (2441, 676)
    # map_filename = "googlemaps-stitch/swx1189609-swy1575631-nex1189639-ney1575606-z22.png"
    # mx, my, zoom = 1189609, 1575606, 22
    # map_filename = "googlemaps-stitch/swx2379218-swy3151263-nex2379279-ney3151213-z23.png"
    # mx, my, zoom = 2379218, 3151213, 23

    ground_plane = Ground(mx, my, zoom, map_filename)
    # lat, lon, alt in meters
    # 50m above home plate
    uav_pos = [40.80164174483064, -77.89348745160423, 10]
    # roll, pitch, yaw, degrees
    uav_att = [25, 0, 0]
    # TODO: add camera attitude

    # this is approximately a gopro hero 3
    # width, height, in pixels, cam resolution
    width, height = 1920, 1080
    draw_width, draw_height = width//2, height//2
    fx, fy = 472, 476
    cx, cy = 960, 540
    # TODO: distortion

    fov_x = np.degrees(2*np.arctan(0.5*width/fx))
    fov_y = np.degrees(2*np.arctan(0.5*height/fy))
    print(f"{fov_x = }. {fov_y = }")

    img = cv2.imread(map_filename)
    h, w, c = img.shape
    print(f"{w = } {h  = }")

    # camera extrinsics
    t = ground_plane.lla_to_pixel(*uav_pos)
    print(f"pixel = {t}")
    t[0] = w/2-t[0]
    t[1] = h/2-t[1]
    T = np.eye(4)
    T[:3,3] = t

    roll, pitch, yaw = uav_att
    theta, phi, gamma = np.radians(-pitch), np.radians(-roll), np.radians(-yaw)
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
    print(R)

    roll, pitch, yaw = uav_att
    c1, s1 = cosd(-yaw), sind(-yaw)
    c2, s2 = cosd(-roll), sind(-roll)
    c3, s3 = cosd(-pitch), sind(-pitch)
    rot_mtx = np.array([[c1*c2, c1*s2*s3-s1*c3, c1*c3*s2+s1*s3]
                        ,[s1*c2, s1*s2*s3+c1*c3, s1*s2*c3-c1*s3]
                        ,[-s2, c2*s3, c2*c3]])
    print(rot_mtx)
    # R = rot_mtx

    # Projection 2D -> 3D matrix
    A1 = np.array([ [1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0, 1],
                    [0, 0, 1]])

    # Projection 3D -> 2D matrix
    A2 = np.array([ [fx, 0, width/2, 0],
                    [0, fy, height/2, 0],
                    [0, 0, 1, 0]])
    
    # extrinsics = np.dot(A2, np.dot(T, np.dot(R, A1)))
    extrinsics = np.dot(A2, np.dot(R, np.dot(T, A1)))
    print(extrinsics)

    out = cv2.warpPerspective(img, extrinsics, (width, height), flags=cv2.INTER_LINEAR)
    # img = cv2.resize(img, shape)
    cv2.imwrite("googlemaps-stitch/out.jpg", out)


if __name__ == "__main__":
    main()
