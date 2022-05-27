# A Closer Look at Few-shot Classification

This repo contains the reference source code for the paper [A Closer Look at Few-shot Classification](https://arxiv.org/abs/1904.04232) in International Conference on Learning Representations (ICLR 2019). In this project, we provide a integrated testbed for a detailed empirical study for few-shot classification.


## Citation
If you find our code useful, please consider citing our work using the bibtex:
```
@inproceedings{
chen2019closerfewshot,
title={A Closer Look at Few-shot Classification},
author={Chen, Wei-Yu and Liu, Yen-Cheng and Kira, Zsolt and Wang, Yu-Chiang and  Huang, Jia-Bin},
booktitle={International Conference on Learning Representations},
year={2019}
}
```

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
```python ./train.py --dataset [DATASETNAME] --model [BACKBONENAME] --method [METHODNAME] [--OPTIONARG]```

For example, run `python ./train.py --dataset miniImagenet --model Conv4 --method baseline --train_aug`  
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
```python ./save_features.py --dataset miniImagenet --model Conv4 --method baseline --train_aug```

## Test
* Change the ```checkpoint_dir``` varibale in ```test.py```
* Run
```python ./test.py --dataset miniImagenet --model Conv4 --method baseline --train_aug```

_With the parameter **--generated_img** you can determine the number of generated images to be taken into account._
## Results
* The test results will be recorded in `./record/results.txt`
* For all the pre-computed results, please see `./record/few_shot_exp_figures.xlsx`. This will be helpful for including your own results for a fair comparison.

## References
Our testbed builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:


* Framework, Backbone, Method: Matching Network
https://github.com/facebookresearch/low-shot-shrink-hallucinate 
* Method: Relational Network
https://github.com/floodsung/LearningToCompare_FSL
* Method: MAML
https://github.com/cbfinn/maml  
https://github.com/dragen1860/MAML-Pytorch  
https://github.com/katerakelly/pytorch-maml
* Data Augmentation: IC-GAN
https://github.com/facebookresearch/ic_gan
* Data Augmentation: DALL·E Mini
https://github.com/borisdayma/dalle-mini
