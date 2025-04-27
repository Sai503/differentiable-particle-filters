from picamera2 import Picamera2
from PIL import Image
from mbot_bridge.api import MBot
# import dpf from particle_filter
import numpy as np
import torch
import sys
sys.path.append("../")
from particle_filter.dpf import DPF
# from particle_filter.utils import load_data, noisyfy_data, make_batch_iterator, get_default_hyperparams
import time
import signal
import csv

default_hyperparams = {
            'init_with_true_state': False,
            'learn_odom': False,
            'use_proposer': True,
            'propose_ratio': 0.7,
            'proposer_keep_ratio': 0.15,
            'min_obs_likelihood': 0.004,
        }

output_file = "test_log.csv"
output_log = []
prev_pose = [0, 0, 0]
prev_pred = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
particle_list = None
particle_probs = None

batch = {
    "o": None,
    "l": None,
    "a": None,
    "s": None
}

def setup():
    picam2 = Picamera2()
    picam2.start()
    time.sleep(1)
    # initialize the mbot driver
    my_robot = MBot()
    my_robot.reset_odometry()
    # init the dpf

    model_path = "../models_trained/full_model.pth"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = DPF(**default_hyperparams)
    method.load_state_dict(torch.load(model_path, map_location=device))
    method.to(device)
    return picam2, my_robot, method

def main_loop(picam2, my_robot, model):
    global prev_pose, particle_list, particle_probs, batch
    start_time = time.time()
    image = picam2.capture_image("main")
    image = image.convert("RGB")
    image = image.resize((32, 32), Image.LANCZOS)
    img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    ranges, thetas = my_robot.read_lidar()
    ranges = torch.tensor(ranges).float()
    thetas = torch.tensor(thetas).float()
    lidar_tensor = process_lidar_scan(ranges, thetas)
    odom_pose = my_robot.read_odometry()
    slam_pose = my_robot.read_slam_pose()
    delta_odom = [odom_pose[0] - prev_pose[0], odom_pose[1] - prev_pose[1], odom_pose[2] - prev_pose[2]]
    prev_pose = odom_pose.copy()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)      # [1, 1, 3, 32, 32]
    lidar_tensor = lidar_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 360, 2]
    delta_odom = torch.tensor(delta_odom).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 3]
    slam_pose = torch.tensor(slam_pose).unsqueeze(0).unsqueeze(0).to(device)    # [1, 1, 3]

    # --- Append new data to batch along time dimension ---
    if batch["o"] is None:
        batch["o"] = img_tensor
        batch["l"] = lidar_tensor
        batch["a"] = delta_odom
        batch["s"] = slam_pose
    else:
        batch["o"] = torch.cat([batch["o"], img_tensor], dim=1)
        batch["l"] = torch.cat([batch["l"], lidar_tensor], dim=1)
        batch["a"] = torch.cat([batch["a"], delta_odom], dim=1)
        batch["s"] = torch.cat([batch["s"], slam_pose], dim=1)
    
    # truncate batch to the last 4 timesteps
    num_ts = 4
    if batch["o"].shape[1] > num_ts:
        batch["o"] = batch["o"][:, -num_ts:, :, :, :]
        batch["l"] = batch["l"][:, -num_ts:, :, :]
        batch["a"] = batch["a"][:, -num_ts:, :]
        batch["s"] = batch["s"][:, -num_ts:, :]

    # Debug: print shapes of batch elements
    print(f"DEBUG: batch['o'] shape: {batch['o'].shape}")
    print(f"DEBUG: batch['l'] shape: {batch['l'].shape}")
    print(f"DEBUG: batch['a'] shape: {batch['a'].shape}")
    print(f"DEBUG: batch['s'] shape: {batch['s'].shape}")

    # --- Run the model on the full batch ---
    if particle_list is None or particle_probs is None:
        pred, particle_list, particle_probs = model.predict(batch, 100, return_particles=True)
    else:
        pred, particle_list, particle_probs = model.predict(
            batch, 100, return_particles=True,
            reuse_initial_particles=False,
            initial_particles=particle_list,
            initial_particle_probs=particle_probs
        )

    # --- Fetch the latest prediction at the last timestep ---
    pred_pose = pred[:, -1]  # shape: [B, 3]
    if pred_pose.dim() == 2:
        pred_pose = pred_pose[0]

    # Squeeze particle_list/probs to keep only the last time step for next iteration
    particle_list = particle_list[:, -1].unsqueeze(1)  # [B, 1, N, 3] -> [B, 1, N, 3]
    particle_probs = particle_probs[:, -1].unsqueeze(1)  # [B, 1, N] -> [B, 1, N]

    prev_pred = pred_pose
    log_update = {
        "timestamp": time.time(),
        "odom_pose_x": odom_pose[0],
        "odom_pose_y": odom_pose[1],
        "odom_pose_theta": odom_pose[2],
        "slam_pose_x": slam_pose[0,0,0].item(),
        "slam_pose_y": slam_pose[0,0,1].item(),
        "slam_pose_theta": slam_pose[0,0,2].item(),
        "pred_pose_x": pred_pose[0].item(),
        "pred_pose_y": pred_pose[1].item(),
        "pred_pose_theta": pred_pose[2].item(),
        "odom_delta_x": delta_odom[0,0,0].item(),
        "odom_delta_y": delta_odom[0,0,1].item(),
        "odom_delta_theta": delta_odom[0,0,2].item(),
    }
    output_log.append(log_update)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time taken: {total_time:.2f} seconds")
    print(f"Predicted pose: {pred_pose}, slam pose: {slam_pose}, odom pose: {odom_pose}, delta odom: {delta_odom}, Timestamp: {log_update['timestamp']}")

def cleanup(picam2, my_robot, model):
    picam2.stop()
    picam2.close()
    my_robot.stop()
    # write to csv
    if len(output_log) == 0:
        print("No data to log")
        return
    writer = csv.DictWriter(open(output_file, "w"), fieldnames=output_log[0].keys())
    writer.writeheader()
    for row in output_log:
        writer.writerow(row)
    print(f"Data logged to {output_file}")

def run():
    running = True
    def handle_sigint(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_sigint)

    picam2, my_robot, model = setup()
    try:
        while running:
            time.sleep(0.5)
            main_loop(picam2, my_robot, model)
    finally:
        cleanup(picam2, my_robot, model)


def process_lidar_scan(ranges, thetas):
    """
    Processes raw LiDAR data to:
    1. Ensure 360 rays (one per degree)
    2. Fill missing rays with neighbor averages
    3. Convert to Cartesian coordinates (x,y)
    
    Args:
        ranges: Tensor of range values
        thetas: Tensor of angle values (radians)
    
    Returns:
        Tensor of shape (360, 2) containing (x,y) coordinates
    """
    # Convert angles to degrees and round to nearest integer
    degrees = torch.rad2deg(thetas).round().long() % 360
    
    # Create output tensor for 360 degrees
    output_ranges = torch.full((360,), float('nan'))
    
    # Assign measured ranges to their degree bins
    for deg, r in zip(degrees, ranges):
        output_ranges[deg] = r
    
    # Fill missing degrees with neighbor averages
    for deg in range(360):
        if torch.isnan(output_ranges[deg]):
            # Find nearest valid neighbors
            prev_deg = (deg - 1) % 360
            next_deg = (deg + 1) % 360
            count = 0
            total = 0.0
            
            # Look backward until we find a valid measurement
            for i in range(1, 360):
                check_deg = (deg - i) % 360
                if not torch.isnan(output_ranges[check_deg]):
                    total += output_ranges[check_deg]
                    count += 1
                    break
            
            # Look forward until we find a valid measurement
            for i in range(1, 360):
                check_deg = (deg + i) % 360
                if not torch.isnan(output_ranges[check_deg]):
                    total += output_ranges[check_deg]
                    count += 1
                    break
            
            if count > 0:
                output_ranges[deg] = total / count
    
    # Convert to Cartesian coordinates
    angles = torch.deg2rad(torch.arange(360, dtype=torch.float32))
    x = output_ranges * torch.cos(angles)
    y = output_ranges * torch.sin(angles)
    
    return torch.stack([x, y], dim=-1)  # (360, 2)

if __name__ == "__main__":
    run()