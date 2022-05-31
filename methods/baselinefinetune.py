import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class BaselineFinetune(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_type = "softmax"):
        super(BaselineFinetune, self).__init__( model_func,  n_way, n_support)
        self.loss_type = loss_type

    def set_forward(self,x,n,is_feature = True):
        return self.set_forward_adaptation(x,n,is_feature); #Baseline always do adaptation
 
    def set_forward_adaptation(self,x, n,is_feature = True):
        assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query  = self.parse_feature(x,n,is_feature)


        z_support   = z_support.contiguous().view(self.n_way* self.n_support, -1 )  # n_way x n_support x 1600 -> (n_way*n_support)x1600
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )      # n_way x n_query x 1600 -> (n_way*n_query)x1600

        y_support = torch.from_numpy(np.repeat(range( self.n_way ), self.n_support ))
        y_support = Variable(y_support.cuda())
        generated = [False if i%self.n_way==0 else True for i in range(0, len(y_support))]
        idxs = self.n_way - n 
        idxs_ini = 0
        for i in range(0, len(generated)):
            for idx in range(idxs_ini, idxs):
                generated[idx] = False
            idxs_ini = idxs_ini + self.n_way
            idxs = idxs + self.n_way
            if idxs > len(generated):
                break
        generated = torch.Tensor(generated)
        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':        
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        
        batch_size = 4
        support_size = self.n_way* self.n_support

        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size , batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                g_batch = generated[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
            
        scores = linear_clf(z_query)
        return scores


    def set_forward_loss(self,x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
        

