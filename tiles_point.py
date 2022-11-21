
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

map_filename = "googlemaps-stitch/40.801127286159506,-77.89370867545878-w220m-h180m-z19-swx148701-swy196953-nex148704-ney196950.jpg"
mx, my, zoom = 148701, 196950, 19

tl_lat, tl_lon = inv_mercator(mx, my, zoom)
print(f"{tl_lat}, {tl_lon}")
m_per_pix = resolution(zoom=zoom, lat=tl_lat)
print(f"{m_per_pix = }")

# px, py = 570, 344
# meters_east = px*m_per_pix
# meters_north = -py*m_per_pix  # use NED signs, so take the negative of y
# print(f"{meters_east}, {meters_north}")

# def ned_translate(lat, lon, dN=0, dE=0):
#     dLat = dN/EARTH_RADIUS_M
#     dLon = dE/(EARTH_RADIUS_M*np.cos(np.radians(lat)))
#     newlat = lat + np.degrees(dLat)
#     newlon = lon + np.degrees(dLon)
#     return (newlat, newlon)

# newlat, newlon = ned_translate(tl_lat, tl_lon, dN=meters_north, dE=meters_east)
# print(f"{newlat}, {newlon}")


# lat, lon, alt in meters
uav_pos = [40.80164174483064, -77.89348745160423, 50]
# roll, pitch, yaw, degrees
uav_att = [0, 0, 0]
# TODO: add camera attitude

# this is approximately a gopro hero 3
# width, height, in pixels, cam resolution
res = [1920, 1080]
fx, fy = 472, 476
cx, cy = 960, 540
mtx = np.array([ 
    [fx,    0,      cx],
    [0,     fx,     cy],
    [0,     0,      1]])


# find corners of the image on the earth plane
# check if the corners are above the horizon


