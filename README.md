# AIMHNet: An Attribute-Insensitive Multi-scale Hourglass network for rain streaks and raindrop removal
## Abstract
### Deep learning has made great progress in single image rain removal. Most recent methods improve performance by increasing the depth of the network. To fully extract local and global features while reducing inference time, we propose a top-to-down attribute-insensitive multi-scale hourglass network for rain streaks and raindrops removal. Considering the difference in the size, shape, orientation and density of rain streaks and raindrops, inspired by the performance of hourglass architecture to capture multi-scale features in human pose estimation, we introduce an hourglass module to remove rain streaks and raindrops in a unified framework. This feature extraction module could capture the characteristics of rain streaks and raindrops with different attributes. These stacked hourglass blocks down-sample features and then up-sample them back to the original shape based on discrete wavelet transform (DWT) and inverse discrete wavelet transform (IDWT). We perform five sets of experiments on synthetic and real-world datasets to validate the effectiveness of our network on rain streaks and raindrop removal. The qualitative and quantitative results show that our method is suitable for removing rain streaks and raindrops.
### 
***
## Experiments
### Requirements
* lpips
* pytorch_wavelets
* pytorch_msssim

### Dataset Preparation
#### Please download Rain1200,RainCityscapes and RainDS：
[Rain1200](https://drive.google.com/file/d/1cMXWICiblTsRl1zjN8FizF5hXOpVOJz4/view?usp=sharing)  
[RainCityscapes](https://www.cityscapes-dataset.com/downloads/)  
[RainDS](https://drive.google.com/file/d/12yN6avKi4Tkrnqa3sMUmyyf4FET9npOT/view?usp=sharing)
#### please put datasets in
> data_path
>> trainA  
>> trainB  
>> testA  
>> testB
***
## Usage
### train
#### `python main.py -name Rain1200 -root /data/users/rain_dataset/Rain1200`
### test
#### `python test_metrics.py`
