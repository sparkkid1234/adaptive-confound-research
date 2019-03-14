import numpy as np
from ..utils import Dataset
from .interface import TopicModel
from .tensorflow_impl.prodlda import VAE
from scipy.sparse import issparse
import math
import tensorflow as tf

"""
A topic model from paper: https://arxiv.org/pdf/1703.01488.pdf but adapted to make z predictive of y
and to make the weight matrix of the last layer sparse
"""


def create_minibatch(data, label, batch_size):
    rng = np.random.RandomState(10)

    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs], label[ixs]


def softmax(x1):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x1) / np.sum(np.exp(x1), axis=0)

def normalize(x):
    res = []
    for x1 in x.T:
        res.append(x1/np.sqrt(np.sum(x1**2)))
    return np.array(res).T
def min_max_normalize(x):
    return (x - x.min())/(x.max()-x.min())
class DiscriminativeProdLDA(TopicModel):
    def __init__(self, k, m, model_params=None, log_params=None):
        super().__init__(k, m, model_params, log_params)
        self.clf = None
        # self._build_model()

    def _build_model(self):
        # <!!!> batch_size = full size of dataset to be used for training since we're using full batch
        network_architecture = self.model_params["network_architecture"]
        learning_rate = self.model_params["learning_rate"]
        batch_size = self.model_params["batch_size"]
        gpu_fraction = self.model_params["gpu_fraction"]
        ld = self.model_params["trade_off"]
        loss = self.model_params["loss"]
        self.clf = VAE(
            network_architecture,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gpu_fraction=gpu_fraction,
            ld = ld,
            loss = loss
        )

    def _reinit(self):
        self.end_session()
        # print("Session closed")
        self._build_model()

    def end_session(self):
        if not self.clf.sess._closed:
            self.clf.sess.close()

    def fit(self, d, x_cluster=False,z=False,anchor=False, *args, **kwargs):
        # Copy or is just referencing ok?
        """
        X: shape (m,n) 
        y: shape (m,1) (use reshape((-1,1)) if y is in shape (m,))
        """
        if self.clf:
            self._reinit()
        else:
            self._build_model()
        if x_cluster:
            X = d.X_train_cluster
        else:
            X = d.X
        if z:
            y = d.z.reshape((-1,1))
        else:
            y = d.y
            if not anchor:
                y = y.reshape((-1, 1))
        n_samples = d.X.shape[0]
        batch_size = self.model_params["batch_size"]
        if issparse(X):
            X = X.todense()
        minibatches = create_minibatch(X, y, batch_size)
        #avg_non_dis = 0
        #avg_dis = 0
        for epoch in range(kwargs["training_epochs"]):
            #prev_non_dis = avg_non_dis
            #prev_dis = avg_dis
            avg_cost = 0.
            #avg_non_dis = 0
            #avg_dis = 0
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = next(minibatches)
                # Fit training using batch data
                cost, emb, non_dis, dis = self.clf.partial_fit(batch_xs, batch_ys)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size
                #avg_non_dis += non_dis / n_samples * batch_size
                #avg_dis += dis / n_samples * batch_size
            # Display logs per epoch step
            """
            if prev_non_dis != 0:
                print("Non dis loss % change: {}".format(abs(avg_non_dis - prev_non_dis)/prev_non_dis))
                print("Dis loss % change: {}".format(abs(avg_dis-prev_dis)/prev_dis))
            """
            if (epoch + 1) % kwargs["display_step"] == 0:
                print(
                    "Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)
                )
        """for epoch in range(kwargs['training_epochs']):
            cost,emb = self.clf.partial_fit(X,y)
            if epoch % kwargs['display_step']==0:
                print("Epoch:", '%04d' % (epoch+1), 
                      "cost=", "{:.9f}".format(cost))
        """

    def get_batches(self, d,x_cluster=False):
        batch_size = self.model_params["batch_size"]
        if x_cluster:
            X = d.X_test_cluster
        else:
            X = d.X
            if issparse(X):
                X = X.todense()
        # pad numpy array so it is divisible by batch_size
        n_rows = X.shape[0]
        div_rem = n_rows % batch_size
        if div_rem:
            pad_size = batch_size - div_rem
            X = np.pad(X, ((0,pad_size), (0,0)), 'constant', constant_values=((0,0),(0,0)))
        # iterate over all the batches
        n_batches = X.shape[0] / batch_size
        return np.vsplit(X, n_batches)
    
    """def transform(self, d, *args, **kwargs):
        topic_softmax=[]
        last_index=0
        batch_size = self.model_params['batch_size']
        X = d.X.todense()
        #Drop instances so total num instances % batch_size == 0 (so that topic_prop works)
        #X = X[:(math.floor(X.shape[0]/batch_size)*batch_size)]
        #X = X.todense()
        if X.shape[0] % 32 != 0:
            tmp = np.asmatrix(np.zeros(((math.ceil(X.shape[0]/32)*32)-X.shape[0]+1,X.shape[1])))
            X = np.vstack((X,tmp))
        batch_idx = np.arange(0, X.shape[0], batch_size)
        for i, j in zip(batch_idx, batch_idx[1:]):
            topic_softmax.extend( self.clf.topic_prop(X[i:j]) )
#         for i in range(X.shape[0]):
#             if i!=0:
#                 if i % batch_size == 0:
#                     tmp = X[last_index:i].todense()
#                     last_index=i
#                     prop = self.clf.topic_prop(tmp)
#                     topic_softmax.extend(softmax(prop))
#                 if i+1 == X.shape[0]:
#                     tmp=X[last_index:].todense()
#                     last_index=i
#                     prop = self.clf.topic_prop(tmp)
#                     topic_softmax.extend(softmax(prop))

        return np.array(topic_softmax[:d.X.shape[0]])"""
    
    def transform(self,d):
        topic_softmax = []
        for batch in self.get_batches(d):
            topic_softmax.extend(self.clf.topic_prop(batch))
        # remove transformation for padded instances
        n_rows = d.X.shape[0]
        topic_softmax = np.array(topic_softmax[:n_rows])
        for i in range(topic_softmax.shape[0]):
            topic_softmax[i,:] = softmax(topic_softmax[i,:])
        return topic_softmax
    
    def predict(self, d,x_cluster=False):
        pred = []
        for batch in self.get_batches(d,x_cluster):
            pred.extend(self.clf.predict(batch))
         # remove predictions for padded instances
        n_rows = d.X.shape[0]
        pred = np.array(pred[:n_rows])
        return pred
    
    def evaluate(self, d):
        prediction = self.predict(d).T[0]
        test = d.y
        return len(test[np.where(prediction == test)])/len(test)
    
    def set_lambda(self,ld):
        self.model_params["trade_off"] = ld
        
    def get_weights(self, idx):
        return self.get_session().run(tf.trainable_variables()[idx])

    def get_session(self):
        return self.clf.sess

    def set_weights(self, new_weights):
        self.clf.set_weights(new_weights)

    def get_name(self):
        if self._name is None:
            self._name = "S-ProdLDA({},{},{})".format(self.k, self.m, 1)
        return self._name
