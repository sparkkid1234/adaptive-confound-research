import argparse

import torch
import torch.nn as nn
from types import SimpleNamespace

opt = SimpleNamespace()
opt.max_epoch = 200
# pre-shuffle for cross validation data split
# for preprocessed amazon dataset; set to -1 to use 30000
opt.feature_num = 5000
# labeled domains: if not set, will use default domains for the dataset
opt.domains = ['train']
opt.unlabeled_domains = ['test']
#parser.add_argument('--emb_filename', default='../data/w2v/word2vec.txt')
#parser.add_argument('--kfold', type=int, default=5) # cross-validation (n>=3)
# which data to be used as unlabeled data: train, unlabeled, or both
opt.random_seed = 1
opt.batch_size = 16
opt.learning_rate = 0.0001
opt.D_learning_rate = 0.0001
opt.fix_emb = False
opt.random_emb = False
opt.F_hidden_sizes = [500, 200]

# gr (gradient reversing, NLL loss in the paper), bs (boundary seeking), l2
# in the paper, we did not talk about the BS loss;
# it's nearly equivalent to the GR (NLL) loss
opt.loss = 'gr'
opt.shared_hidden_size = 128
opt.domain_hidden_size = 64
opt.activation = 'relu'
opt.F_layers = 1
opt.C_layers = 3
opt.D_layers = 3
opt.n_critic = 5
opt.lambd = 0.05
# batch normalization
opt.F_bn = False
opt.C_bn = True
opt.D_bn = True
opt.dropout = 0.4
opt.device = 'cuda'
opt.num_labels = 2

# automatically prepared options
if not torch.cuda.is_available():
    opt.device = 'cpu'

if len(opt.domains) == 0:
    # use default domains
    opt.domains = ['train']
    
opt.all_domains = opt.domains + opt.unlabeled_domains

if opt.activation.lower() == 'relu':
    opt.act_unit = nn.ReLU()
elif opt.activation.lower() == 'leaky':
    opt.act_unit = nn.LeakyReLU()
else:
    raise Exception(f'Unknown activation function {opt.activation}')
