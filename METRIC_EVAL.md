# DepthPRo Metric Depth Evaluation Benchmark on Several Datasets
This forked repository contains scripts formatted as test_****.sh, which downloads and runs the model on the dataset. For the curent version, NYU-V2 Depth and KITTI Eigen Split Datasets are available for evaluation. 

## Getting Ready
Running the scripts requires a correctly-set DepthPro environment, which user needs to follow the project's installation guide.
While installation, if environment name is set to anything other than default (depth-pro), it is required to change the environment name in .sh scripts.
If environment name is default, one may easily proceed to run the scripts by:
```bash
bash test_kitti.sh
```
or
```bash
bash test_nyu.sh
```
Equivalently, user may run the python scripts all by themselves as well.


## About the code
Once you run the script, it will try to download the respective dataset under /home/{your_username}/nyu_cache, or a fancy-pathed cache file under kagglehub. Whenever you use any of my scripts once again, it will refer to the previously downloaded dataset.

It is further possible to change the dataset sampling for NYUV2 by:  
```python
dataset = load_dataset("sayakpaul/nyu_depth_v2", split="validation[:654]", cache_dir=home_dir+"/nyu_cache")
dataset = dataset.select(range(0, 654, 6))  # Sample every 6th data in dataset
```


## Acknowledgment
This work is based on (and forked from) [DepthPro](https://github.com/apple/ml-depth-pro), developed by [Apple](https://github.com/apple).    
NYUV2 Dataset used can be found at [here](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2). KITTI Eigen Split is taken from [here](https://www.kaggle.com/datasets/awsaf49/kitti-eigen-split-dataset), with some modifications to test_locations.txt.

