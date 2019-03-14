import numpy as np
from ..utils import Dataset
from .interface import TopicModel
from .man_model.layers import *
from .man_model.models import *
from .man_model import utils
import torch.utils.data as data_utils

def create_minibatch(data, label,z, batch_size):
    rng = np.random.RandomState(10)

    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs], label[ixs], z[ixs]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / np.sum(e_x, axis=1).reshape(-1, 1)

class MultinomialAdversarialNetwork(TopicModel):
    def __init__(self, k, m, domains, model_params=None, log_params=None):
        super().__init__(k,m,model_params,log_params)
        self.domains = domains
        self.loss = 'gr'
        self.clf = None
        
    def prepare_data(self,d):
        """
        Assume d is a dictionary of dataset where d[domain] = another dataset class
        Assume labeled domain = train set, unlabeled = test
        """
        train_loaders, train_iters = {}, {}
        for domain in self.domains:
            features, target = torch.from_numpy(d[domain].X.todense()), torch.from_numpy(d[domain].y)
            train = data_utils.TensorDataset(features,target)
            train_loaders[domain] = DataLoader(train, self.model_params['batch_size'], shuffle = True)
            train_iters[domain] = iter(train_loaders[domain])
        return train_loaders, train_iters
            
            
    def fit(self, d, *args, **kwargs):
        X = d.X
        y = d.y.reshape((-1, 1))
        z = d.z.reshape((-1,1))
        n_samples = d.X.shape[0]
        batch_size = self.model_params["batch_size"]
        if issparse(X):
            X = X.todense()
        #minibatches = create_minibatch(X, y, z, batch_size)
        train_loaders, train_iters = self.prepare_data(d)
        
        #Training
        F_s = MlpFeatureExtractor(X.shape[1], [1000,500],128, 0.4)
        F_d = {}
        for domain in self.domains:
            F_d[domain] = MlpFeatureExtractor(X.shape[1], [1000,500],128, 0.4)
            
        C = SentimentClassifier(3, 128+64,128+64, 1,0.4, True)
        D = DomainClassifier(3, 128, 128,len(self.domains), 'gr', 0.4, True)

        F_s, C, D = F_s.to('cuda'), C.to('cuda'), D.to('cuda')
        for f_d in F_d.values():
            f_d = f_d.to('cuda')
        # optimizers
        optimizer = optim.Adam(itertools.chain(*map(list, [F_s.parameters() if F_s else [], C.parameters()] + [f.parameters() for f in F_d.values()])), lr=0.0001)
        optimizerD = optim.Adam(D.parameters(), lr=0.0001)
        # training
        best_acc, best_avg_acc = defaultdict(float), 0.0

        for epoch in range(kwargs["training_epochs"]):
            F_s.train()
            C.train()
            D.train()
            for f in F_d.values():
                f.train()

            # training accuracy
            correct, total = defaultdict(int), defaultdict(int)
            # D accuracy
            d_correct, d_total = 0, 0
            # conceptually view 1 epoch as 1 epoch of the first domain
            num_iter = len(train_loaders[self.domains[0]])
            for i in tqdm(range(num_iter)):
                # D iterations
                utils.freeze_net(F_s)
                map(utils.freeze_net, F_d.values())
                utils.freeze_net(C)
                utils.unfreeze_net(D)
                # optional WGAN n_critic trick
                n_critic = 5

                for _ in range(n_critic):
                    D.zero_grad()
                    loss_d = {}
                    # train on both labeled and unlabeled domains
                    for domain in self.domains:
                        # targets not used
                        #Thay unlabeled_loaders/iters = train_loaders/iters?
                        d_inputs, _, d_targets = utils.endless_get_next_batch(
                            train_loaders, train_iters, domain)
                        shared_feat = F_s(d_inputs)
                        d_outputs = D(shared_feat)
                        # D accuracy
                        _, pred = torch.max(d_outputs, 1)
                        d_total += len(d_inputs)
                        if self.loss.lower() == 'l2':
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
                utils.unfreeze_net(F_s)
                map(utils.unfreeze_net, F_d.values())
                utils.unfreeze_net(C)
                utils.freeze_net(D)
                #if opt.fix_emb:
                #    utils.freeze_net(F_s.word_emb)
                #    map(utils.freeze_net, F_d.values())
                F_s.zero_grad()
                for f_d in F_d.values():
                    f_d.zero_grad()
                C.zero_grad()
                shared_feats, domain_feats = [], []
                for domain in self.domains:
                    inputs, targets = utils.endless_get_next_batch(
                            train_loaders, train_iters, domain)
                    targets = targets.to('cuda')
                    shared_feat = F_s(inputs)
                    shared_feats.append(shared_feat)
                    domain_feat = F_d[domain](inputs)
                    domain_feats.append(domain_feat)
                    features = torch.cat((shared_feat, domain_feat), dim=1)
                    c_outputs = C(features)
                    l_c = functional.nll_loss(c_outputs, targets)
                    l_c.backward(retain_graph=True)
                    # training accuracy
                    _, pred = torch.max(c_outputs, 1)
                    total[domain] += targets.size(0)
                    correct[domain] += (pred == targets).sum().item()
                # update F with D gradients on all domains
                for domain in self.domains:
                    d_inputs, _ = utils.endless_get_next_batch(
                            unlabeled_loaders, unlabeled_iters, domain)
                    shared_feat = F_s(d_inputs)
                    d_outputs = D(shared_feat)
                    if self.loss.lower() == 'gr':
                        d_targets = utils.get_domain_label(self.loss, domain, len(d_inputs))
                        l_d = functional.nll_loss(d_outputs, d_targets)
                        log.debug(f'D loss: {l_d.item()}')
                        if opt.lambd > 0:
                            l_d *= -opt.lambd
                    elif self.loss.lower() == 'l2':
                        d_targets = utils.get_random_domain_label(opt.loss, len(d_inputs))
                        l_d = functional.mse_loss(d_outputs, d_targets)
                        if opt.lambd > 0:
                            l_d *= opt.lambd
                    l_d.backward()

                optimizer.step()

            # end of epoch
            log.info('Ending epoch {}'.format(epoch+1))