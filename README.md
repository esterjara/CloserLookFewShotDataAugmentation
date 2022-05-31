# Synthetic Few Shot Learning
This repo contains the reference source code for the paper "When a Word is Worth a Thousand Images: Visual Learning with
Natural-Language-based Data Augmentation". In this project, we provide a test set for an empirical study of few-shot classification with data augmentation focused on the IC-GAN model and the DALL-E neural network.


## Enviroment
 - Python3
 - [Pytorch](http://pytorch.org/) before 0.4 (for newer vesion, please see issue #3 )
 - json

## Getting started

### mini-ImageNet
* Change directory to `./filelists/miniImagenet`
* run `source ./download_miniImagenet.sh` 

(WARNING: This would download the 155G ImageNet dataset. You can comment out correponded line 5-6 in `download_miniImagenet.sh` if you already have one.) 

## Train
Run
```python train.py --dataset [DATASETNAME] --model [BACKBONENAME] --method [METHODNAME] [--OPTIONARG]```

For example, run `python train.py --dataset miniImagenet --model Conv4 --method baseline --train_aug`  
Commands below follow this example, and please refer to io_utils.py for additional options.

## Data Augmentation
#### IC-GAN
* Change directory to ```cd IC-GAN ```
* Download all necessary libraries ```python install.py```
* Generate images from the IC-GAN model ```python icgan.py```
* Run ```generated_images.py```
#### DALL·E Mini
* Change directory to ```cd DALLE ```
* Download all necessary libraries ```python install.py```
* Generate images from the DALL·E neural network ```python dalle.py```
* Run ```generated_images_dalle.py```

## Save features
Save the extracted feature before the classifaction layer to increase test speed. This is not applicable to MAML, but are required for other methods.
* Modify the variables ```generated_loadfile``` and ```classes_file``` in the ```save_features.py``` script with the JSON files generated above. 
* Change the ```checkpoint_dir``` varibale in ```save_features.py```
* Run
```python save_features.py --dataset miniImagenet --model Conv4 --method baseline --train_aug```

## Test
* Change the ```checkpoint_dir``` varibale in ```test.py```
* Run
```python test.py --dataset miniImagenet --model Conv4 --method baseline --train_aug```

_With the parameter **--generated_img** you can determine the number of generated images to be taken into account._
## Results
* The test results will be recorded in `./record/results.txt`
* For all the pre-computed results, please see `./record/few_shot_exp_figures.xlsx`. This will be helpful for including your own results for a fair comparison.

## References
Our testbed builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:

* Framework: https://github.com/wyharveychen/CloserLookFewShot
* Backbone, Method: Matching Network
https://github.com/facebookresearch/low-shot-shrink-hallucinate 
* Data Augmentation: IC-GAN
https://github.com/facebookresearch/ic_gan
* Data Augmentation: DALL·E Mini
https://github.com/borisdayma/dalle-mini
