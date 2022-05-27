#!/usr/bin/env python3
"""
IC-GAN: Instance-Conditioned GAN

https://github.com/facebookresearch/ic_gan, Copyright (c) 2021 Facebook
"""
__author__ = "Ester Jara Lorente"
__since__ = "2022/03/27"

import os
import sys
import torch
import imageio
import torchvision
import json
import nltk
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

torch.manual_seed(np.random.randint(sys.maxsize))
from torch import nn
from time import time
from IPython.display import HTML, Image, clear_output
from PIL import Image
from base64 import b64encode
from nltk.corpus import wordnet as wn
from scipy.stats import truncnorm, dirichlet
from pytorch_pretrained_biggan import BigGAN, convert_to_images, one_hot_from_names, utils

sys.path.append('./ic_gan/inference')
sys.path[0] = './ic_gan/inference'
sys.path.insert(1, os.path.join(sys.path[0], ".."))
torch.manual_seed(np.random.randint(sys.maxsize))
import cma
from cma.sigma_adaptation import CMAAdaptSigmaCSA, CMAAdaptSigmaTPA
import warnings
warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)
import torchvision.transforms as transforms
import inference.utils as inference_utils
import data_utils.utils as data_utils
from BigGAN_PyTorch.BigGAN import Generator as generator
import sklearn.metrics


def replace_to_inplace_relu(model): #saves memory; from https://github.com/minyoungg/pix2latent/blob/master/pix2latent/model/biggan.py
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU(inplace=False))
        else:
            replace_to_inplace_relu(child)
    return
    
def save(out,name=None, torch_format=True):
  if torch_format:
    with torch.no_grad():
      out = out.cpu().numpy()
  img = convert_to_images(out)[0]
  if name:
    imageio.imwrite(name, np.asarray(img))
  return img

hist = []
def checkin(i, best_ind, total_losses, losses, regs, out, noise=None, emb=None, probs=None):
  global sample_num, hist
  name = None
  if save_every and i%save_every==0:
    name = './output/frame_%05d.jpg'%sample_num
  pil_image = save(out, name)
  vals0 = [sample_num, i, total_losses[best_ind], losses[best_ind], regs[best_ind], np.mean(total_losses), np.mean(losses), np.mean(regs), np.std(total_losses), np.std(losses), np.std(regs)]
  stats = 'sample=%d iter=%d best: total=%.2f cos=%.2f reg=%.3f avg: total=%.2f cos=%.2f reg=%.3f std: total=%.2f cos=%.2f reg=%.3f'%tuple(vals0)
  vals1 = []
  if noise is not None:
    vals1 = [np.mean(noise), np.std(noise)]
    stats += ' noise: avg=%.2f std=%.3f'%tuple(vals1)
  vals2 = []
  if emb is not None:
    vals2 = [emb.mean(),emb.std()]
    stats += ' emb: avg=%.2f std=%.3f'%tuple(vals2)
  elif probs:
    best = probs[best_ind]
    inds = np.argsort(best)[::-1]
    probs = np.array(probs)
    vals2 = [ind2name[inds[0]], best[inds[0]], ind2name[inds[1]], best[inds[1]], ind2name[inds[2]], best[inds[2]], np.sum(probs >= 0.5)/pop_size,np.sum(probs >= 0.3)/pop_size,np.sum(probs >= 0.1)/pop_size]
    stats += ' 1st=%s(%.2f) 2nd=%s(%.2f) 3rd=%s(%.2f) components: >=0.5:%.0f, >=0.3:%.0f, >=0.1:%.0f'%tuple(vals2)
  hist.append(vals0+vals1+vals2)
  if show_every and i%show_every==0:
    clear_output()
    display(pil_image)  
  sample_num += 1

def load_icgan(experiment_name, root_ = './'):
  root = os.path.join(root_, experiment_name)
  config = torch.load("%s/%s.pth" %
                      (root, "state_dict_best0"))['config']

  config["weights_root"] = root_
  config["model_backbone"] = 'biggan'
  config["experiment_name"] = experiment_name
  G, config = inference_utils.load_model_inference(config)
  G.cuda()
  G.eval()
  return G

def get_output(noise_vector, input_label, input_features, model, channels):  
  if stochastic_truncation: #https://arxiv.org/abs/1702.04782
    with torch.no_grad():
      trunc_indices = noise_vector.abs() > 2*truncation
      size = torch.count_nonzero(trunc_indices).cpu().numpy()
      trunc = truncnorm.rvs(-2*truncation, 2*truncation, size=(1,size)).astype(np.float32)
      noise_vector.data[trunc_indices] = torch.tensor(trunc, requires_grad=requires_grad, device='cuda')
  else:
    noise_vector = noise_vector.clamp(-2*truncation, 2*truncation)
  if input_label is not None:
    input_label = to

  else:
    input_label = None

  out = model(noise_vector, input_label.cuda() if input_label is not None else None, input_features.cuda() if input_features is not None else None)
  
  if channels==1:
    out = out.mean(dim=1, keepdim=True)
    out = out.repeat(1,3,1,1)
  return out

def normality_loss(vec): #https://arxiv.org/abs/1903.00925
    mu2 = vec.mean().square()
    sigma2 = vec.var()
    return mu2+sigma2-torch.log(sigma2)-1
    

def load_generative_model(gen_model, last_gen_model, experiment_name, model):
  # Load generative model
  if gen_model != last_gen_model:
    model = load_icgan(experiment_name, root_ = './')
    last_gen_model = gen_model
  return model, last_gen_model

def load_feature_extractor(gen_model, last_feature_extractor, feature_extractor):
  # Load feature ext
  feat_ext_name = 'classification' if gen_model == 'cc_icgan' else 'selfsupervised'
  if last_feature_extractor != feat_ext_name:
    if feat_ext_name == 'classification':
      feat_ext_path = ''
    else:
      # !curl -L -o ./swav_pretrained.pth.tar -C - 'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar' 
      feat_ext_path = './swav_pretrained.pth.tar'
    last_feature_extractor = feat_ext_name
    feature_extractor = data_utils.load_pretrained_feature_extractor(feat_ext_path, feature_extractor = feat_ext_name)
    feature_extractor.eval()
  return feature_extractor, last_feature_extractor

norm_mean = torch.Tensor([0.485, 0.456, 0.406])#.view(3, 1, 1)
norm_std = torch.Tensor([0.229, 0.224, 0.225])#.view(3, 1, 1)

def preprocess_input_image(input_image_path, size): 
  pil_image = Image.open(input_image_path).convert('RGB')
  transform_list =  transforms.Compose([data_utils.CenterCropLongEdge(), transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
  transform_list =  transforms.Compose([data_utils.CenterCropLongEdge(), transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
  tensor_image = transform_list(pil_image)
  tensor_image = torch.nn.functional.interpolate(tensor_image.unsqueeze(0), 224, mode="bicubic", align_corners=True)
  return tensor_image

def preprocess_generated_image(image): 
  # transform_list =  transforms.Normalize(norm_mean, norm_std)
  # image = transform_list(image*0.5 + 0.5)
  image = torch.nn.functional.interpolate(image, 224, mode="bicubic", align_corners=True)
  return image

stochastic_truncation = False
truncation =  0.7

last_gen_model = None
last_feature_extractor = None
model = None
feature_extractor = None

def image_generator(label, path):

  gen_model = 'icgan'  
  experiment_name = 'icgan_biggan_imagenet_res256'

  size = '256'
  input_feature_index = 3
  num_samples_ranked = 30
  num_samples_total = 100
  truncation =  0.7
  seed =  50

  if seed == 0:
      seed = None
  noise_size = 128
  class_size = 1000
  channels = 3
  batch_size = 4

  last_gen_model = None
  last_feature_extractor = None
  model = None
  feature_extractor = None

  if gen_model == 'icgan':
      class_index = None

  assert(num_samples_ranked <=num_samples_total)
  state = None if not seed else np.random.RandomState(seed)
  np.random.seed(seed)

  feature_extractor_name = 'classification' if gen_model == 'cc_icgan' else 'selfsupervised'
  os.makedirs("../img", exist_ok=True)
  os.makedirs("../img/"+str(label), exist_ok=True)

  # Path of the original image
  input_image_instance = path
  # Load feature extractor (outlier filtering and optionally input image feature extraction)
  feature_extractor, last_feature_extractor = load_feature_extractor(gen_model, last_feature_extractor, feature_extractor)
  # Load features 
  if input_image_instance not in ['None', ""]:
      input_feature_index = None
      input_image_tensor = preprocess_input_image(input_image_instance, int(size))
      print(input_image_tensor.shape)
      with torch.no_grad():
          input_features, _ = feature_extractor(input_image_tensor.cuda())
      input_features/=torch.linalg.norm(input_features,dim=-1, keepdims=True)
  elif input_feature_index is not None:
      input_features = np.load('./stored_instances/imagenet_res'+str(size)+'_rn50_'+feature_extractor_name+'_kmeans_k1000_instance_features.npy', allow_pickle=True).item()["instance_features"][input_feature_index:input_feature_index+1]
  else:
      input_features = None

  # Load generative model
  model, last_gen_model = load_generative_model(gen_model, last_gen_model, experiment_name, model)

  # Prepare other variables
  name_file = '%s_class_index%s_instance_index%s'%(gen_model, str(class_index) if class_index is not None else 'None', str(input_feature_index) if input_feature_index is not None else 'None')
  os.makedirs("../outputs", exist_ok=True)

  replace_to_inplace_relu(model)
  ind2name = {index: wn.of2ss('%08dn'%offset).lemma_names()[0] for offset, index in utils.IMAGENET.items()}

  eps = 1e-8

  # Create noise, instance and class vector
  noise_vector = truncnorm.rvs(-2*truncation, 2*truncation, size=(num_samples_total, noise_size), random_state=state).astype(np.float32) #see https://github.com/tensorflow/hub/issues/214
  noise_vector = torch.tensor(noise_vector, requires_grad=False, device='cuda')
  if input_features is not None:
    instance_vector = torch.tensor(input_features, requires_grad=False, device='cuda').repeat(num_samples_total, 1)
  else: 
    instance_vector = None
  if class_index is not None:
    input_label = torch.LongTensor([class_index]*num_samples_total)
  else:
    input_label = None

  size = int(size)
  all_outs, all_dists = [], []
  for i_bs in range(num_samples_total//batch_size+1):
      start = i_bs*batch_size
      end = min(start+batch_size, num_samples_total)
      if start == end:
          break

      
      out = get_output(noise_vector[start:end], input_label[start:end] if input_label is not None else None, instance_vector[start:end] if instance_vector is not None else None, model, channels)

      if instance_vector is not None:
          # Get features from generated images + feature extractor
          out_ = preprocess_generated_image(out)
          with torch.no_grad():
              out_features, _ = feature_extractor(out_.cuda())
          out_features/=torch.linalg.norm(out_features,dim=-1, keepdims=True)
          dists = sklearn.metrics.pairwise_distances(
                  out_features.cpu(), instance_vector[start:end].cpu(), metric="euclidean", n_jobs=-1)
          all_dists.append(np.diagonal(dists))
          all_outs.append(out.detach().cpu())
      del (out)
  all_outs = torch.cat(all_outs)
  all_dists = np.concatenate(all_dists)

  # Order samples by distance to conditioning feature vector and select only num_samples_ranked images
  selected_idxs =np.argsort(all_dists)[:num_samples_ranked]
  row_i, col_i, i_im = 0, 0, 0
  all_images_mosaic = np.zeros((3,size*(int(np.sqrt(num_samples_ranked))), size*(int(np.sqrt(num_samples_ranked)))))
  paths = []
  for k,j in enumerate(selected_idxs):
    path = '../img/'+str(label)+'/image_'+str(label)+'_'+str(k)+'.png'
    image = save(all_outs[j][np.newaxis,...], path, torch_format=False)
    paths.append(path)
  return paths
  
if __name__ == "__main__":
  sys.path.append('./CLIP')
  import clip

  last_clip_model = 'ViT-B/32'
  perceptor, preprocess = clip.load(last_clip_model)

  nltk.download('wordnet')
  nltk.download('omw-1.4')

  from pytorch_pretrained_biggan import BigGAN, convert_to_images, one_hot_from_names, utils
  os.chdir('./')

  # Path to the original images used as input for the IC-GAN
  novel_path = '/mnt/colab_public/projects/pau/closer_look/filelists/miniImagenet/novel.json'
  with open(novel_path, 'r') as f:
            novel_images = json.load(f)

  dic_samples = {}
  for path in novel_images['image_names']:
    path_list = path.split('/')
    label = path_list[-1][:-5]
    input_image_instance = '/mnt/colab_public/datasets/Imagenet/raw/train/'+str(path_list[-2])+'/'+str(path_list[-1])
    print(input_image_instance)
    generated_images = image_generator(label, input_image_instance)
    dic_samples[input_image_instance] = generated_images

  with open('/mnt/home/CloserLookFewShot/samples.json', 'w') as fp:
    json.dump(dic_samples, fp)