import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import backbone
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 

@torch.no_grad()
def save_features(model, data_loader, outfile, n_shot ):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_feats = f.create_dataset('all_feats', (max_count, n_shot, 1600), dtype='f')
    all_labels = f.create_dataset('all_labels',(max_count,n_shot), dtype='S10')
    all_targets = f.create_dataset('all_targets',(max_count,n_shot), dtype='i')
    count=0

    for i, (x,y,z) in enumerate(data_loader):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x = x.squeeze(0)
        x_var = Variable(x)     # [n, 3, 84, 84]
        feats = model(x_var)    # nx1600
        all_feats[i] = feats.cpu()
        targets = torch.flatten(torch.stack(y))
        all_targets[i] = targets
        all_labels[i] = [item.encode("ascii", "ignore") for lab in z for item in lab]
        count += 1
        
    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

if __name__ == '__main__':
    # Change the "PATH" variable
    PATH = '/mnt/home/CloserLookFewShotDataAugmentation/'
    params = parse_args('save_features')
    assert params.method != 'maml' and params.method != 'maml_approx', 'maml do not support save_feature and run'

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84 
    else:
        image_size = 224

    loadfile = PATH+'novel.json'
    if params.data_aug == 'icgan':
        generated_loadfile = PATH+'generated_images.json'
        classes_file = PATH+'samples_dalle.json'

    else:
        generated_loadfile = PATH+'generated_images_dalle.json'
        classes_file = PATH+'samples_dalle.json'
   
    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    split = params.split
    if params.dataset == 'cross':
        if split == 'base':
            loadfile = configs.data_dir['miniImagenet'] + 'all.json' 
        else:
            loadfile   = configs.data_dir['CUB'] + split +'.json' 
    elif params.dataset == 'cross_char':
        if split == 'base':
            loadfile = configs.data_dir['omniglot'] + 'noLatin.json' 
        else:
            loadfile  = configs.data_dir['emnist'] + split +'.json' 
    else:
        loadfile = configs.data_dir[params.dataset] + split + '.json'


    if params.method == 'baseline':
        checkpoint_dir = '/mnt/colab_public/projects/pau/closer_look/checkpoints/miniImagenet/Conv4_baseline_aug'
    else:
        checkpoint_dir = '/mnt/colab_public/projects/pau/closer_look/checkpoints/miniImagenet/Conv4_baseline++_aug'


    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
   else:
        modelfile   = get_best_file(checkpoint_dir)

    if params.save_iter != -1:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5") 
    else:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5") 


    datamgr         = SimpleDataManager(image_size, batch_size = 1)         # batch_size=64
    data_loader      = datamgr.get_data_loader(loadfile, generated_loadfile, classes_file, aug = False)

    if params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4': 
            model = backbone.Conv4NP()
        elif params.model == 'Conv6': 
            model = backbone.Conv6NP()
        elif params.model == 'Conv4S': 
            model = backbone.Conv4SNP()
        else:
            model = model_dict[params.model]( flatten = False )
    elif params.method in ['maml' , 'maml_approx']: 
       raise ValueError('MAML do not support save feature')
    else:
        model = model_dict[params.model]()

    model = model.cuda()
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model.load_state_dict(state, strict=False)
    model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, data_loader, outfile, params.n_shot)
