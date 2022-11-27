# https://github.com/doersino/aerialbot

import io
import math
import os
import re
import requests
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from multiprocessing.dummy import Pool as ThreadPool

def main():
    TILE_SIZE = 256  # pixels
    EARTH_CIRCUMFERENCE = 40075.016686 * 1000  # in meters, at the equator   
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36"
    tile_path_template = "googlemaps-tiles/z{zoom}x{x}y{y}.png"
    tile_url_template = "https://khms2.google.com/kh/v={google_maps_version}?x={x}&y={y}&z={zoom}"
    image_path_template = "googlemaps-stitch/swx{swx}-swy{swy}-nex{nex}-ney{ney}-z{zoom}.png"

    # inputs
    lat = 40.801127286159506
    lon = -77.89370867545878
    geowidth = 220  # meters, longitude range
    geoheight = 180  # meters, latitude range
    meters_per_pixel = 0.0127  # meters/pixel, sames as 0.5 in/pixel
    image_quality = 50

    # Setup the output folders
    out_folder_tiles = "googlemaps-tiles"
    out_folder_stitch = "googlemaps-stitch"
    os.makedirs(out_folder_tiles, exist_ok=True)
    os.makedirs(out_folder_stitch, exist_ok=True)

    # Get the latest google maps version
    print("Determining current Google Maps version and patching tile URL template...")
    # automatic fallback: current as of October 2021, will likely continue to work for at least a while
    google_maps_version = '908'
    try:
        google_maps_page = requests.get("https://maps.googleapis.com/maps/api/js", headers={"User-Agent": USER_AGENT}).content
        match = re.search(rb'null,\[\[\"https:\/\/khms0\.googleapis\.com\/kh\?v=([0-9]+)', google_maps_page)
        if match:
            google_maps_version = match.group(1).decode('ascii')
            print(f"Using google maps version: {google_maps_version}")
        else:
            print(f"Unable to extract current version, proceeding with outdated version {google_maps_version} instead.")
    except requests.RequestException:
        print(f"Unable to load Google Maps, proceeding with outdated version {google_maps_version} instead.")

    meters_per_pixel_at_zoom_0 = ((EARTH_CIRCUMFERENCE / TILE_SIZE) * math.cos(math.radians(lat)))
    desired_zoom = np.log(meters_per_pixel_at_zoom_0/meters_per_pixel)/np.log(2)
    print(f"Desired zoom for {meters_per_pixel} m/pix: {desired_zoom}")
    zoom = int(np.clip(desired_zoom, 0, 23))  # Max zoom is 23
    # zoom = 23
    print(f"Using zoom: {zoom}")
    in_per_m = 39.3701
    print(f"GSD: {meters_per_pixel_at_zoom_0/(2**zoom):.4f} m/pix. {(2**zoom)/(in_per_m*meters_per_pixel_at_zoom_0):.4f} pix/in.")

    # Find the southwest and northeast corners
    meters_per_degree = (EARTH_CIRCUMFERENCE / 360)
    width_geo = geowidth / (meters_per_degree * math.cos(math.radians(lat)))
    height_geo = geoheight / meters_per_degree
    sw_lat, sw_lon = (lat - height_geo / 2, lon - width_geo / 2)
    ne_lat, ne_lon = (lat + height_geo / 2, lon + width_geo / 2)
    print(f"{sw_lat=} {sw_lon=}")
    print(f"{ne_lat=} {ne_lon=}")
    EARTH_RADIUS = EARTH_CIRCUMFERENCE / (1000 * 2 * math.pi)
    spherical_cap_difference = (2 * math.pi * EARTH_RADIUS ** 2) * abs(math.sin(math.radians(sw_lat)) - math.sin(math.radians(ne_lat)))
    area = spherical_cap_difference * (ne_lon - sw_lon) / 360
    print(f"{area=} km^2")

    # Convert the southwest and northeast corners to map tiles xyz
    # Use WebMercator projection
    def WebMercator_projection(lat, lon, zoom):
        factor = (1 / (2 * math.pi)) * 2 ** zoom
        x = factor * (math.radians(lon) + math.pi)
        y = factor * (math.pi - math.log(math.tan((math.pi / 4) + (math.radians(lat) / 2))))
        return (math.floor(x), math.floor(y))
    swx, swy = WebMercator_projection(sw_lat, sw_lon, zoom)
    nex, ney = WebMercator_projection(ne_lat, ne_lon, zoom)
    print(f"{swx=} {swy=}")
    print(f"{nex=} {ney=}")
    # Include the boundary tiles with +1
    tiles_height = swy-ney+1
    tiles_width = nex-swx+1
    total_tiles = tiles_width*tiles_height
    print(f"Width = {tiles_width}. Height = {tiles_height}")
    print(f"Total tiles: {total_tiles}")

    t1 = time.time()
    missing_tiles = []  # tuples
    image_tiles = [[None for i in range(tiles_height)] for j in range(tiles_width)]
    def download_image(tile):
        x, y, zoom = tile
        idx, idy = x-swx, y-ney
        # Check if the tile is donwloaded first
        filename = tile_path_template.format(zoom=zoom, x=x, y=y)
        # print(f"{filename = }")
        if os.path.isfile(filename):
            # print(f"Already got: {x, y, zoom}")
            current_image = Image.open(filename)
            image_tiles[idx][idy] = current_image  # add good image
        else:
            # Request the tile from googlemaps
            try:
                url = tile_url_template.format(
                    google_maps_version=google_maps_version,
                    x=x, y=y, zoom=zoom
                )
                # print(f"{url = }")
                r = requests.get(url, headers={'User-Agent': USER_AGENT})
                # Convert response into an image
                data = r.content
                current_image = Image.open(io.BytesIO(data))
                assert current_image.mode == "RGB"
                assert current_image.size == (TILE_SIZE, TILE_SIZE)
                current_image.save(filename)
                image_tiles[idx][idy] = current_image  # add good image
            except Exception as e:
                print(e)
                # print(f"Failed to download: {current_tile}")
                missing_tiles.append((x, y, zoom))
                print(f"{url = }")
                exit()
    
    all_tiles = []
    for x in range(swx, nex+1):
        for y in range(ney, swy+1):
            all_tiles.append((x, y, zoom))

    threads = 24
    pool = ThreadPool(threads)
    pool.map(download_image, all_tiles)

    dt = time.time()-t1
    # Report tiles that failed and are not loaded
    if len(missing_tiles)>0:
        print(f"Missing tiles: {missing_tiles}")
    print(f"Download complete! {dt:.2f}s. {1000*dt/(tiles_width*tiles_height):.1f} ms/img.")

    print("Stitching tiles")
    # Stitch the tiles together, ignore failed tiles
    stitch_filename =  image_path_template.format(
        swx=swx, swy=swy,
        nex=nex, ney=ney,
        zoom=zoom,
    )
    stitch = Image.new('RGB', (tiles_width * TILE_SIZE, tiles_height * TILE_SIZE))
    for x in range(0, tiles_width):
        for y in range(0, tiles_height):
            if image_tiles[x][y] is not None:
                stitch.paste(image_tiles[x][y], (x * TILE_SIZE, y * TILE_SIZE))
    stitch.save(stitch_filename)

    print("Complete")


if __name__ == "__main__":
    main()
