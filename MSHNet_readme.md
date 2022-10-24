# Multi-scale Stacked Hourglass network for single image rain removal
## Abstract
### 
***
## Experiments
### Requirements
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