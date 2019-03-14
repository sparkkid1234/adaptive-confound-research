#This version assumes domains = train/test set
import numpy as np
from ..utils import Dataset
import math
import random
from .interface import TopicModel
from .man_model.models import *
from .man_model import utils
from .man_model.options import opt
import torch.utils.data as data_utils
from tqdm import tqdm
from collections import defaultdict
import itertools
from torchnet.meter import ConfusionMeter

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

"""
IMPORTANT: for some reason, Model (self.F_s,etc) will not work if inputs are not float32
=> need to convert. Dont know if same thing for target tho?
Also apparently, domain labels retrieved from get_domain_labels cannot be -1?
Output size for C HAS TO BE 2 even if it's a binary classification
"""
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / np.sum(e_x, axis=1).reshape(-1, 1)

class MultinomialAdversarialNetwork(TopicModel):
    def __init__(self, k, m, model_params=None, log_params=None):
        super().__init__(k,m,model_params,log_params)
        
    def prepare_data(self,d):
        """
        Assume d is a dictionary of dataset where d[domain] = another dataset class
        Assume labeled domain = train set, unlabeled = test
        """
        train_loaders, train_iters = {}, {}
        unlabeled_loaders, unlabeled_iters = {}, {}
        for domain in opt.domains:
            #CONVERT TO FLOAT32
            features, target = torch.from_numpy(d[domain].X.todense().astype('float32')), torch.from_numpy(d[domain].y)#.reshape((-1,1))
            train = data_utils.TensorDataset(features,target)
            train_loaders[domain] = DataLoader(train, opt.batch_size, shuffle = True)
            train_iters[domain] = iter(train_loaders[domain])
        for domain in opt.unlabeled_domains:
            features, target = torch.from_numpy(d[domain].X.todense().astype('float32')), torch.from_numpy(d[domain].y)#.reshape(-1,1))
            uset = data_utils.TensorDataset(features,target)
            unlabeled_loaders[domain] = DataLoader(uset,opt.batch_size, shuffle = True)
            unlabeled_iters[domain] = iter(unlabeled_loaders[domain])
               
        return train_loaders, train_iters, unlabeled_loaders, unlabeled_iters
            
            
    def fit(self, d, *args, **kwargs):
        #minibatches = create_minibatch(X, y, z, batch_size)
        #TODO: make this able to fit consecutively
        train_loaders, train_iters, unlabeled_loaders, unlabeled_iters = self.prepare_data(d)
        #Training
        self.F_s = MlpFeatureExtractor(d['train'].X.shape[1], opt.F_hidden_sizes,opt.shared_hidden_size, opt.dropout)
        self.F_d = {}
        for domain in opt.domains:
            self.F_d[domain] = MlpFeatureExtractor(d['train'].X.shape[1], opt.F_hidden_sizes, opt.domain_hidden_size, opt.dropout)
        self.C = SentimentClassifier(opt.C_layers, opt.shared_hidden_size + opt.domain_hidden_size, opt.shared_hidden_size + opt.domain_hidden_size, 2,opt.dropout, opt.C_bn)
        self.D = DomainClassifier(opt.D_layers, opt.shared_hidden_size, opt.shared_hidden_size,len(opt.all_domains), opt.loss, opt.dropout, opt.D_bn)
#         print("try")
#         print(opt.device)
        self.F_s, self.C, self.D = self.F_s.to(opt.device), self.C.to(opt.device), self.D.to(opt.device)
        for f_d in self.F_d.values():
            f_d = f_d.to(opt.device)
#         print("endtry")
#         # optimizers
        optimizer = optim.Adam(itertools.chain(*map(list, [self.F_s.parameters() if self.F_s else [], self.C.parameters()] + [f.parameters() for f in self.F_d.values()])), lr=0.0001)
        optimizerD = optim.Adam(self.D.parameters(), lr=0.0001)
        loss_d_res = []
        l_d_res = []
        l_c_res = []
        for epoch in range(opt.max_epoch):
            self.F_s.train()
            self.C.train()
            self.D.train()
            for f in self.F_d.values():
                f.train()

            # training accuracy
            correct, total = defaultdict(int), defaultdict(int)
            # D accuracy
            d_correct, d_total = 0, 0
            # conceptually view 1 epoch as 1 epoch of the first domain
            num_iter = len(train_loaders[opt.domains[0]])
            for i in range(num_iter):
                # D iterations
                utils.freeze_net(self.F_s)
                map(utils.freeze_net, self.F_d.values())
                utils.freeze_net(self.C)
                utils.unfreeze_net(self.D)
                # optional WGAN n_critic trick
                n_critic = opt.n_critic

                for _ in range(n_critic):
                    self.D.zero_grad()
                    loss_d = {}
                    # train on both labeled and unlabeled domains
                    for domain in opt.unlabeled_domains:
                        # targets not used
                        d_inputs, _ = utils.endless_get_next_batch(
                            unlabeled_loaders, unlabeled_iters, domain)
                        d_inputs = d_inputs.to(opt.device)
                        d_targets = utils.get_domain_label(opt.loss, domain, len(d_inputs))
                        shared_feat = self.F_s(d_inputs)
                        d_outputs = self.D(shared_feat)
                        # D accuracy
                        _, pred = torch.max(d_outputs, 1)
                        d_total += len(d_inputs)
                        if opt.loss.lower() == 'l2':
                            _, tgt_indices = torch.max(d_targets, 1)
                            d_correct += (pred==tgt_indices).sum().item()
                            l_d = functional.mse_loss(d_outputs, d_targets)
                            l_d.backward()
                        else:
                            d_correct += (pred==d_targets).sum().item()
                            l_d = functional.nll_loss(d_outputs, d_targets)
                            l_d.backward()
                        loss_d[domain] = l_d.item()
                    optimizerD.step()
                # F&C iteration
                utils.unfreeze_net(self.F_s)
                map(utils.unfreeze_net, self.F_d.values())
                utils.unfreeze_net(self.C)
                utils.freeze_net(self.D)
                #if opt.fix_emb:
                #    utils.freeze_net(self.F_s.word_emb)
                #    map(utils.freeze_net, self.F_d.values())
                self.F_s.zero_grad()
                for f_d in self.F_d.values():
                    f_d.zero_grad()
                self.C.zero_grad()
                shared_feats, domain_feats = [], []
                for domain in opt.domains:
                    inputs, targets = utils.endless_get_next_batch(
                            train_loaders, train_iters, domain)
                    #target = torch.int64 rn
                    targets = targets.to(opt.device)
                    inputs = inputs.to(opt.device)
                    shared_feat = self.F_s(inputs)
                    shared_feats.append(shared_feat)
                    domain_feat = self.F_d[domain](inputs)
                    domain_feats.append(domain_feat)
                    features = torch.cat((shared_feat, domain_feat), dim=1)
                    c_outputs = self.C(features)
                    #return c_outputs, targets
                    #DEVICE SIDE TRIGGERED ERROR OCCUR HERE (l_c=...)
                    l_c = functional.nll_loss(c_outputs, targets)
                    l_c.backward(retain_graph=True)
                    # training accuracy
                    _, pred = torch.max(c_outputs, 1)
                    total[domain] += targets.size(0)
                    correct[domain] += (pred == targets).sum().item()
                # update F with D gradients on all domains
                for domain in opt.unlabeled_domains:
                    d_inputs, _ = utils.endless_get_next_batch(
                            unlabeled_loaders, unlabeled_iters, domain)
                    d_inputs = d_inputs.to(opt.device)
                    shared_feat = self.F_s(d_inputs)
                    d_outputs = self.D(shared_feat)
                    if opt.loss.lower() == 'gr':
                        d_targets = utils.get_domain_label(opt.loss, domain, len(d_inputs))
                        l_d = functional.nll_loss(d_outputs, d_targets)
                        if opt.lambd > 0:
                            l_d *= -opt.lambd
                    elif opt.loss.lower() == 'l2':
                        d_targets = utils.get_random_domain_label(opt.loss, len(d_inputs))
                        l_d = functional.mse_loss(d_outputs, d_targets)
                        if opt.lambd > 0:
                            l_d *= opt.lambd
                    l_d.backward()
                

                optimizer.step()
            

#             print(loss_d)
#             print('l_d loss: {}'.format(l_d.item()))
#             print('l_c loss: {}'.format(l_c.item()))
            loss_d_res.append(loss_d['test'])
            l_d_res.append(l_d.item())
            l_c_res.append(l_c.item())
            if (epoch + 1) % kwargs["display_step"] == 0:
                print(
                    "Epoch:", "%04d, done" % (epoch + 1) #"cost=", "{:.9f}"#.format(l_d.data[0])
                )
        return loss_d_res, l_d_res, l_c_res
    
    def transform(self, d, *args, **kwargs):
        F_d = self.F_d[opt.domains[0]]
        self.F_s.eval()
        F_d.eval()
        self.C.eval()
        _,_,_,it = self.prepare_data(d)
        it = it[opt.unlabeled_domains[0]]
        correct = 0
        total = 0
        confusion = ConfusionMeter(opt.num_labels)
        preds = []
        for inputs,targets in it:
            inputs = inputs.to(opt.device)
            targets = targets.to(opt.device)
            d_features = F_d(inputs)
            features = torch.cat((self.F_s(inputs), d_features), dim=1)
            outputs = self.C(features)
            _, pred = torch.max(outputs, 1)
            #preds.extend(pred.data)
            confusion.add(pred.data, targets.data)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
        acc = correct / total
        #('{}: Accuracy on {} samples: {}%'.format(name, total, 100.0*acc))
        return acc, correct
        #return preds
    
    def get_name(self):
        if self._name is None:
            self._name = "MAN({},{},{})".format(self.k,self.m,1)
        return self._name