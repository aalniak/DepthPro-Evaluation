# Depth Anything V2 Metric Depth Evaluation Benchmark on Several Datasets
This forked repository contains scripts under ./metric_depth folder, which downloads and runs the model on the dataset. For the curent version, NYU-V2 Depth Dataset and KITTI Eigen Split Dataset are available for evaluation. 

## Getting Ready
Running the scripts requires a correctly-set Depth-Anything-V2 environment, which user follows the original project's installation guide. Further, to use metric depth estimation models, it is a must to download and put the fine-tuned [models](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md#pre-trained-models) under ./metric_depth/checkpoints.

While installation, if environment name is set to anything other than default (Depth-Anything-V2), it is a requirement to change the environment name from .sh scripts.
If environment name is default, one may easily proceed to run the scripts by:
```bash
bash kitti_test.sh
```
or
```bash
bash nyu_test.sh
```
Equivalently, user may run the python scripts all by themselves as well.

## Parameters

It worths noting that the .sh script gives the model some priori information, such as max_depth, and which model to use... There are two suggested models, that are Hypersim (for indoor with suggested max_depth=20) and VKITTI (for outdoor with suggested max_depth=80). 

For KITTI Eigen Split evalutaion, script variables are set as follows:    
input_size = 518,  
model=Fine-tuned on VKITTI,    
max_depth=80.  


For NYU-V2 Depth Evaluation, script variables are as follows:
input_size = 518,  
model=Fine-tuned on Hypersim,    
max_depth=20.  


## About the code
Once you run the script, it will try to download the respective dataset under /home/{your_username}/nyu_cache, or a fancy-pathed cache file under kagglehub. Whenever you use any of my scripts once again, it will refer to the previously downloaded dataset.

It is further possible to change the dataset sampling for NYUV2 by:  
```python
dataset = load_dataset("sayakpaul/nyu_depth_v2", split="validation[:654]", cache_dir=home_dir+"/nyu_cache")
dataset = dataset.select(range(0, 654, 6))  # Sample every 6th data in dataset
```


## Acknowledgment
This work is based on [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2), developed by [Binyi Kang](https://github.com/bingykang).    
NYUV2 Dataset used can be found at [here](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2). KITTI Eigen Split is taken from [here](https://www.kaggle.com/datasets/awsaf49/kitti-eigen-split-dataset), with some modifications to test_locations.txt.

