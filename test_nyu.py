import logging
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
from depth_pro import create_model_and_transforms, load_rgb
import os

home_dir = os.environ["HOME"] # to save the nyu_cache
print(home_dir)

LOGGER = logging.getLogger(__name__)

def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

def compute_depth_metrics(pred_depth, gt_depth):
    """Compute depth estimation error metrics: RMSE, log RMSE, AbsRel, SqRel, and SI Log Error."""
    valid_mask = gt_depth > 0  # Avoid invalid depth values
    pred_depth = pred_depth[valid_mask]
    gt_depth = gt_depth[valid_mask]
    
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

def process_dataset(dataset,model, transform):
    total_metrics = {"RMSE": 0, "Log RMSE": 0, "AbsRel": 0, "SqRel": 0, "SI Log Error": 0}
    num_samples = 0
    
    for idx, sample in enumerate(list(dataset)):
        print(f"Processing Sample {idx}")
        rgb_image = sample['image']  # RGB image
        gt_depth = np.array(sample['depth_map'])  # Ground truth depth
        inferred_depth = run_test(rgb_image, model, transform)
        
        metrics = compute_depth_metrics(inferred_depth, gt_depth)
        
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        num_samples += 1
        print(f"Unique ID (Scene Name): {sample.get('scene', 'Unknown')}")
        print(f"Depth Map Hash: {hash(gt_depth.tobytes())}")  # Ensure unique depth values
        print(f"Metrics: {metrics}")
    
    avg_metrics = {key: total_metrics[key] / num_samples for key in total_metrics}
    print(f"Average Metrics: {avg_metrics}")
    return avg_metrics

def main():

    # Load model.
    model, transform = create_model_and_transforms(
    device=get_torch_device(),
    precision=torch.half,   
    )
    model.eval()
    dataset = load_dataset("sayakpaul/nyu_depth_v2", split="validation[:654]", cache_dir=home_dir+"/nyu_cache")
    #dataset = dataset.select(range(0, 654, 6))  # Sample every 6th data in dataset
    process_dataset(dataset, model, transform)
            


if __name__ == "__main__":
    main()