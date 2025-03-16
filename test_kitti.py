import kagglehub
import logging
import torch
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
from depth_pro import create_model_and_transforms, load_rgb
import argparse
import os
# KITTI Depth Scaling Factor (as per official documentation)
DEPTH_SCALE = 256.0  
MAX_DEPTH = 80  # Maximum depth for occluded/missing pixels

LOGGER = logging.getLogger(__name__)

def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

def compute_depth_metrics(pred_depth, gt_depth, max_depth=MAX_DEPTH):
    """Compute depth estimation error metrics, ignoring MAX_DEPTH pixels."""
    valid_mask = (gt_depth > 0) & (gt_depth < max_depth)  # Ignore zero and MAX_DEPTH pixels

    
    pred_depth = pred_depth[valid_mask]
    gt_depth = gt_depth[valid_mask]

    if len(gt_depth) == 0:
        print("Warning: No valid pixels found for metric computation!")
        return {"RMSE": np.nan, "Log RMSE": np.nan, "AbsRel": np.nan, "SqRel": np.nan, "SI Log Error": np.nan}

    # RMSE
    rmse = np.sqrt(mean_squared_error(gt_depth, pred_depth))

    # Log RMSE
    log_rmse = np.sqrt(mean_squared_error(np.log(gt_depth), np.log(pred_depth)))

    # Absolute Relative Difference
    absrel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)

    # Squared Relative Difference
    sqrel = np.mean(((gt_depth - pred_depth) ** 2) / gt_depth)

    # Scale Invariant Log Error
    log_diff = np.log(pred_depth) - np.log(gt_depth)
    silog = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))

    return {
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "AbsRel": absrel,
        "SqRel": sqrel,
        "SI Log Error": silog
    }


def run_test(img,model,transform):
    """Run Depth Pro on a sample image."""
        # Load image and focal length from exif info (if found.).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_px = None

    # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
    # otherwise the model estimates `f_px` to compute the depth metricness.
    #torch.tensor([5017.97], dtype=torch.float32, device=device
    prediction = model.infer(transform(img), f_px=f_px) #f_px is valuable

    # Extract the depth and focal length.
    depth = prediction["depth"].detach().cpu().numpy().squeeze()

    if f_px is not None:
        LOGGER.debug(f"Focal length (from exif): {f_px:0.2f}")
    elif prediction["focallength_px"] is not None:
        focallength_px = prediction["focallength_px"].detach().cpu().item()
        LOGGER.info(f"Estimated focal length: {focallength_px}")
    return depth

def run_kitti_test(dataset_path,model,transform):
    total_metrics = {"RMSE": 0, "Log RMSE": 0, "AbsRel": 0, "SqRel": 0, "SI Log Error": 0}
    num_samples = 0
    
    dataset_path += "/"
    print("Path to dataset files:", dataset_path)
    
    # Read file containing image-depth pairs
    with open(dataset_path + "test_locations.txt", "r") as file:
        for line in file:
            img_path, gt_path = line.strip().split(" ")

            # Load depth map as uint16 (as per KITTI format)
            gt_depth = cv2.imread(dataset_path + gt_path, cv2.IMREAD_UNCHANGED)
            if gt_depth is None:
                print(f"Error loading depth map: {dataset_path + gt_path}")
                continue
            # Convert depth map to float32 and scale to meters
            gt_depth = gt_depth.astype(np.float32) / DEPTH_SCALE
            # Create a mask for invalid pixels (zero values = no valid depth data)
            zero_mask = (gt_depth == 0)
            # Set invalid depth pixels to MAX_DEPTH for visualization purposes
            gt_depth[zero_mask] = MAX_DEPTH
            rgb_image = cv2.imread(dataset_path + img_path, cv2.IMREAD_UNCHANGED)
            rgb_image = np.array(rgb_image)
            inferred_depth = run_test(rgb_image, model, transform)
            
        
            metrics = compute_depth_metrics(inferred_depth, gt_depth, MAX_DEPTH)

            # Accumulate metrics
            for key in total_metrics:
                total_metrics[key] += metrics[key]
        
            num_samples += 1
        
            print(f"Metrics: {metrics}")
            
    
    # Compute average metrics
        avg_metrics = {key: total_metrics[key] / num_samples for key in total_metrics}
        print(f"Average Metrics: {avg_metrics}")
        return avg_metrics

#Load model.
model, transform = create_model_and_transforms(
device=get_torch_device(),
precision=torch.half,   
)
model.eval()


        

# Download the dataset from KaggleHub
dataset_path = kagglehub.dataset_download("aalniak/kitti-eigen-split-for-monocular-depth-estimation")
# Call the function to visualize depth maps correctly
run_kitti_test(dataset_path,model,transform)
