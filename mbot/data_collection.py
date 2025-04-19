from mbot_bridge.api import MBot
import time
import csv
import os
import subprocess

my_robot = MBot()

my_robot.reset_odometry()

path = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 1.57],
    [0.0, 1.0, 3.14],
    [0.0, 0.0, -1.57]
]
#this is a line I have added
my_robot.drive_path(path)

# Ensure directories exist
os.makedirs("images", exist_ok=True)

with open("lidar_scans.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "timestamp", "range", "theta", 
        "image_filename", 
        "odom_x", "odom_y", "odom_theta",
        "slam_x", "slam_y", "slam_theta"
    ])

    start_time = time.time()
    duration = 10  # seconds to scan
    scan_interval = 0.5  # seconds between scans

    while time.time() - start_time < duration:
        timestamp = time.time()
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        image_filename = f"images/{ts_str}.jpg"

        # Capture image
        subprocess.run(["libcamera-jpeg", "-o", image_filename, "--timeout", "1"], check=True)

        # Read LiDAR
        ranges, thetas = my_robot.read_lidar()

        # Read odometry and SLAM poses
        odom_pose = my_robot.read_odometry()      # returns (x, y, theta)
        slam_pose = my_robot.read_slam_pose()     # returns (x, y, theta)

        # Save LiDAR rays with poses
        for r, theta in zip(ranges, thetas):
            if r > 0.0:
                writer.writerow([
                    timestamp, r, theta, image_filename,
                    odom_pose[0], odom_pose[1], odom_pose[2],
                    slam_pose[0], slam_pose[1], slam_pose[2]
                ])

        time.sleep(scan_interval)
