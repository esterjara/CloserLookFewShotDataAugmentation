# Synthetic Few Shot Learning
This repo contains the reference source code for the paper "When a Word is Worth a Thousand Images: Visual Learning with
Natural-Language-based Data Augmentation". In this project, we provide a test set for an empirical study of the meta-augmentation framework, whereby the addition of randomness given by new images improves generalization to new tasks.
In particular, two completely different data augmentation approaches are considered: (i) IC-GAN, which generates synthetic images from an original image as input; (ii) DALL·E, which generates plausible images from text captions.
## Enviroment
 - Python3
 - [Pytorch](http://pytorch.org/) before 0.4 (for newer vesion, please see issue #3 )
 - json

## Getting started

### mini-ImageNet
* Change directory to `./filelists/miniImagenet`
* run `source ./download_miniImagenet.sh` 

(WARNING: This would download the 155G ImageNet dataset. You can comment out correponded line 5-6 in `download_miniImagenet.sh` if you already have one.) 

### CIFAR-100
Two Google Colab notebooks are provided, one to generate images with IC-GAN and the other with DALL-E Mini. Then this meta-augmentation is used for the few-shot classification.


## Train
Run
```python train.py --dataset [DATASETNAME] --model [BACKBONENAME] --method [METHODNAME] [--OPTIONARG]```

For example, run `python train.py --dataset miniImagenet --model Conv4 --method baseline --train_aug`  
Commands below follow this example, and please refer to io_utils.py for additional options.

## Data Augmentation
Creates the folder where all the necessary json files will be stored: `mkdir json`
#### IC-GAN
* Change directory to ```cd IC-GAN ```
* Generate images from the IC-GAN model ```python icgan.py```
* Run ```generated_images.py```
#### DALL·E Mini
* Change directory to ```cd DALLE ```
* Generate images from the DALL·E neural network ```python dalle.py```
* Run ```generated_images_dalle.py```

(WARNING: The images that will be generated correspond to the novel classes of the mini-ImageNet dataset).

## Save features
Save the extracted feature before the classifaction layer to increase test speed. This is not applicable to MAML, but are required for other methods.
* Modify the variable `$PATH` with the path to the previously created json folder 
* Change the ```checkpoint_dir``` variable
* Run
```python save_features.py --dataset miniImagenet --model Conv4 --method baseline --train_aug```
#### Additional and optional parameters:
* `split`: base/val/novel
* `n_shot`: Number of labeled data in each class
* `data_aug`: icgan or dalle
* `save_iter`: Save feature from the model trained in x epoch, use the best model if x is -1

## Test
* Change the ```checkpoint_dir``` varibale in ```test.py```
* Run
```python test.py --dataset miniImagenet --model Conv4 --method baseline --train_aug```
#### Additional and optional parameters:
* `split`: base/val/novel
* `n_shot`: Number of labeled data in each class
* `generated_img`: Number of synthetic images used
* `save_iter`: Saved feature from the model trained in x epoch, use the best model if x is -1

## Results
The test results will be recorded in `./record/results.txt`

## References
Our testbed builds upon several existing publicly available code. Specifically, we have modified and integrated the following code into this project:

* Framework: https://github.com/wyharveychen/CloserLookFewShot
* Backbone, Method: Matching Network
https://github.com/facebookresearch/low-shot-shrink-hallucinate 
* Data Augmentation: IC-GAN
https://github.com/facebookresearch/ic_gan
* Data Augmentation: DALL·E Mini
https://github.com/borisdayma/dalle-mini
