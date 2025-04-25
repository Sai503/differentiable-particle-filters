import csv
import numpy as np
import os
from PIL import Image
import math

csv_file = "test_log.csv"
map_file = "map.csv"

output_file = "final.png"
pixel_colors = {
    "odom": (255, 0, 0),  # Red
    "slam": (0, 255, 0),  # Green
    "pred": (0, 0, 255),  # Blue
}
origin_x, origin_y, cells_x, cells_y, meters_per_cell = None, None, None, None, None

def coordinate_to_pixel(x, y):
    global origin_x, origin_y, meters_per_cell
    pixel_x = math.floor((x - origin_x) / meters_per_cell)
    pixel_y = math.floor((y - origin_y) / meters_per_cell)
    return pixel_x, pixel_y

def main():
    # open map file
    with open(map_file, mode="r") as file:
        # read first line
        header = file.readline().strip().split(" ")
        origin_x, origin_y, cells_x, cells_y, meters_per_cell = header
        origin_x = float(origin_x)
        origin_y = float(origin_y)
        cells_x = int(cells_x)
        cells_y = int(cells_y)
        meters_per_cell = float(meters_per_cell)
        # read rest of map as numpy array
        map_data = np.loadtxt(file, delimiter=" ", dtype=int)
        # scale map data from -127 to 127 to 0 to 255
        map_data = (map_data + 127)
        # invert map data
        map_data = 255 - map_data
        # convert to grayscale image
        map_image = Image.fromarray(map_data.astype(np.uint8), mode="L")
        # convert to RGB image
        map_image = map_image.convert("RGB")

        # open csv log file with dict reader
        with open(csv_file, mode="r") as file2:
            reader = csv.DictReader(file2)
            # iterate over rows in csv file
            for row in reader:
                # get odom, slam, and pred poses
                odom_x = float(row["odom_x"])
                odom_y = float(row["odom_y"])
                slam_x = float(row["slam_x"])
                slam_y = float(row["slam_y"])
                pred_x = float(row["pred_x"])
                pred_y = float(row["pred_y"])

                # convert to pixel coordinates
                odom_pixel_x, odom_pixel_y = coordinate_to_pixel(odom_x, odom_y)
                slam_pixel_x, slam_pixel_y = coordinate_to_pixel(slam_x, slam_y)
                pred_pixel_x, pred_pixel_y = coordinate_to_pixel(pred_x, pred_y)

                # draw pixels on map image
                map_image.putpixel((odom_pixel_x, odom_pixel_y), pixel_colors["odom"])
                map_image.putpixel((slam_pixel_x, slam_pixel_y), pixel_colors["slam"])
                map_image.putpixel((pred_pixel_x, pred_pixel_y), pixel_colors["pred"])

            # save the map image
            map_image.save(output_file)
            print(f"Map image saved as {output_file}")

        


if __name__ == "__main__":
    main()