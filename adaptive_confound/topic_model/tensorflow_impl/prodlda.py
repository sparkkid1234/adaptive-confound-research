import numpy as np
import tensorflow as tf
import numpy as np
import itertools, time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import pickle
import time


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform(
        (fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32
    )


def log_dir_init(fan_in, fan_out, topics=50):
    return tf.log((1.0 / topics) * tf.ones([fan_in, fan_out]))


class VAE(object):
    """
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(
        self,
        network_architecture,
        transfer_fct=tf.nn.softplus,
        learning_rate=0.001,
        batch_size=100,
        gpu_fraction=0.25,
        ld = 1.0,
        loss = 'cross_entropy'
    ):
        tf.reset_default_graph()
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.ld = ld
        print("Learning Rate:", self.learning_rate)
        self.drop_out = 1
        self.loss = loss
        # Regularizer to be applied to any or all of the layers
        self.l1 = tf.contrib.layers.l1_regularizer(scale=0.005)
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        self.y = tf.placeholder(tf.float32, [None, self.network_architecture["n_output"]])
        self.keep_prob = tf.placeholder(tf.float32)

        self.h_dim = float(network_architecture["n_z"])
        self.a = 1 * np.ones((1, int(self.h_dim))).astype(np.float32)
        self.mu2 = tf.constant((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)
        self.var2 = tf.constant(
            (
                ((1.0 / self.a) * (1 - (2.0 / self.h_dim))).T
                + (1.0 / (self.h_dim * self.h_dim)) * np.sum(1.0 / self.a, 1)
            ).T
        )

        self._create_network()
        self._create_loss_optimizer()

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        config.gpu_options.allocator_type = "BFC"
        self.sess = tf.Session(config=config)
        self.sess.run(init)

    def _create_network(self):
        self.network_weights = self._initialize_weights(**self.network_architecture)
        self.z_mean, self.z_log_sigma_sq = self._recognition_network(
            self.network_weights["weights_recog"], self.network_weights["biases_recog"]
        )
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32,seed = 42)
        self.z = tf.add(
            self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps)
        )
        self.sigma = tf.exp(self.z_log_sigma_sq)
         
        self.x_reconstr_mean, self.y_pred = self._generator_network(
            self.z, self.network_weights["weights_gener"]
        )

    def _initialize_weights(
        self,
        n_hidden_recog_1,
        n_hidden_recog_2,
        n_hidden_recog_3,
        n_hidden_gener_1,
        n_input,
        n_z,
        n_output,
    ):
        all_weights = dict()
        all_weights['weights_recog'] = {
            #Add l1 regularizer to weight matrix of 1st layer
            'h1': tf.get_variable('h1',[n_input, n_hidden_recog_1]),#,regularizer = self.l1),
            'h2': tf.get_variable('h2',[n_hidden_recog_1, n_hidden_recog_2]),#,regularizer = self.l1),
            'h3': tf.get_variable('h3',[n_hidden_recog_2, n_hidden_recog_3]),#,regularizer = self.l1),
            #'h4': tf.get_variable('h4',[n_hidden_recog_3, n_hidden_recog_4]),
            'out_mean': tf.get_variable('out_mean',[n_hidden_recog_3, n_z]),#,regularizer = self.l1),
            'out_log_sigma': tf.get_variable('out_log_sigma',[n_hidden_recog_3, n_z])}
            #'out': tf.Variable(tf.zeros([n_z,n_output],dtype=tf.float32))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'b3': tf.Variable(tf.zeros([n_hidden_recog_3], dtype=tf.float32)),
            #'b4': tf.Variable(tf.zeros([n_hidden_recog_4], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
            #'b4': tf.Variable(tf.zeros([n_output],dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h2': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'out': tf.Variable(tf.zeros([n_z,n_output],dtype=tf.float32)),
            'b1': tf.Variable(tf.zeros([n_output],dtype=tf.float32))}

        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network)
        layer_1 = self.transfer_fct(
            tf.add(tf.matmul(self.x, weights["h1"]), biases["b1"])
        )
        layer_2 = self.transfer_fct(
            tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
        )
        layer_3 = self.transfer_fct(
            tf.add(tf.matmul(layer_2, weights["h3"]), biases["b3"])
        )
        if self.drop_out == 0:
            z_mean = tf.contrib.layers.batch_norm(
                tf.add(tf.matmul(layer_3, weights["out_mean"]), biases["out_mean"])
            )
            z_log_sigma_sq = tf.contrib.layers.batch_norm(
                tf.add(
                    tf.matmul(layer_3, weights["out_log_sigma"]),
                    biases["out_log_sigma"],
                )
            )
#             eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), 0, 1, dtype=tf.float32)
#             z = tf.add(
#                 z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)
#             )
#             y_pred = tf.add(tf.matmul(tf.nn.softmax(z), weights["out"]), biases["b4"])
#             self.predictions = tf.round(tf.sigmoid(y_pred))
        else:
            layer_do = tf.nn.dropout(layer_3, self.keep_prob)
            z_mean = tf.contrib.layers.batch_norm(
                tf.add(tf.matmul(layer_do, weights["out_mean"]), biases["out_mean"])
            )
            z_log_sigma_sq = tf.contrib.layers.batch_norm(
                tf.add(
                    tf.matmul(layer_do, weights["out_log_sigma"]),
                    biases["out_log_sigma"],
                )
            )
#             eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), 0, 1, dtype=tf.float32)
#             z = tf.add(
#                 z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)
#             )
#             self.layer_do_0 = tf.nn.dropout(tf.nn.softmax(z), self.keep_prob)
#             y_pred = tf.add(tf.matmul(self.layer_do_0, weights["out"]), biases["b4"])
#             self.predictions = tf.round(tf.sigmoid(y_pred))
            

        return (z_mean, z_log_sigma_sq)#,y_pred)

    def _generator_network(self, z, weights):
        if self.drop_out == 0:
            y_pred = tf.add(tf.matmul(tf.nn.softmax(z), weights["out"]), weights["b1"])
            self.predictions = tf.round(tf.sigmoid(y_pred))
            x_reconstr_mean = tf.nn.softmax(
                tf.contrib.layers.batch_norm(
                    tf.add(tf.matmul(tf.nn.softmax(z), weights["h2"]), 0.0)
                )
            )
        else:
            self.layer_do_0 = tf.nn.dropout(tf.nn.softmax(z), self.keep_prob)
            y_pred = tf.add(tf.matmul(self.layer_do_0, weights["out"]), weights["b1"])
            self.predictions = tf.round(tf.sigmoid(y_pred))
            x_reconstr_mean = tf.nn.softmax(
                tf.contrib.layers.batch_norm(
                    tf.add(tf.matmul(self.layer_do_0, weights["h2"]), 0.0)
                )
            )
        return x_reconstr_mean, y_pred

    def _create_loss_optimizer(self):
        self.x_reconstr_mean += 1e-10
        
        reconstr_loss = -tf.reduce_sum(
            self.x * tf.log(self.x_reconstr_mean), 1
        )  # /tf.reduce_sum(self.x,1)
        if self.loss == 'cross_entropy':
            discriminative_loss = tf.reduce_sum(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_pred, labels=self.y)
            )
        elif self.loss == 'mse':
            discriminative_loss = tf.reduce_sum(tf.losses.mean_squared_error(self.y,self.y_pred))
            
        latent_loss = 0.5 * (
            tf.reduce_sum(tf.div(self.sigma, self.var2), 1)
            + tf.reduce_sum(
                tf.multiply(
                    tf.div((self.mu2 - self.z_mean), self.var2),
                    (self.mu2 - self.z_mean),
                ),
                1,
            )
            - self.h_dim
            + tf.reduce_sum(tf.log(self.var2), 1)
            - tf.reduce_sum(self.z_log_sigma_sq, 1)
        )

        # Regularizer (only regularize first layer making it sparse)
        #reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # print(reg)
        #reg_const = 0.01
        self.non_dis = tf.reduce_mean(reconstr_loss) + tf.reduce_mean(latent_loss)
        self.dis_loss = discriminative_loss
        self.cost = (
            (tf.reduce_mean(reconstr_loss)
            + tf.reduce_mean(latent_loss))
            + (1/self.ld * tf.reduce_mean(discriminative_loss))
            #+ tf.reduce_mean(reg_const * sum(reg))
        )  # average over batch

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=0.99
        ).minimize(self.cost)
    
    def partial_fit(self, X, y):
        opt, cost, emb, non_dis,dis = self.sess.run(
            (self.optimizer, self.cost, self.network_weights["weights_gener"]["h2"],
            self.non_dis, self.dis_loss),
            feed_dict={self.x: X, self.y: y, self.keep_prob: .4},
        )
        return cost, emb, non_dis , dis

    def test(self, X):
        """Test the model and return the lowerbound on the log-likelihood.
        """
        cost = self.sess.run(
            (self.cost),
            feed_dict={self.x: np.expand_dims(X, axis=0), self.keep_prob: 1.0},
        )
        return cost

    def predict(self, X):
        #Apply sigmoid and rounding before use
        if self.drop_out == 1:
            pred =  self.sess.run((self.predictions),feed_dict={self.x: X,self.keep_prob:0.4})
        else:
            pred = self.sess.run((self.predictions),feed_dict={self.x:X})
        return pred
    
    def get_discriminative_loss(self, X,y):
        return self.sess.run((self.discriminative_loss),feed_dict={self.x:X,self.y:y})
    
    def get_total_loss(self, X):
        return self.sess.run((self.cost),feed_dict={self.x:X})
        
    def set_weights(self, new_weights):
        """
        Use placeholder to be able to call set_weights consecutively (else sess.run returns only a tensor and thus)
        """
        feed_dict = {}
        tmp = self.network_weights["weights_gener"]["out"]
        new_weights = np.asarray(new_weights, dtype=np.float32)
        tf_dtype = tf.as_dtype(tmp.dtype.name.split("_")[0])
        if hasattr(tmp, "_assign_placeholder"):
            assign_placeholder = self.network_weights["weights_gener"][
                "out"
            ]._assign_placeholder
            assign_op = self.network_weights["weights_gener"]["out"]._assign_op
        else:
            assign_placeholder = tf.placeholder(tf_dtype, shape=new_weights.shape)
            assign_op = self.network_weights["weights_gener"]["out"].assign(
                assign_placeholder
            )
            self.network_weights["weights_gener"][
                "out"
            ]._assign_placeholder = assign_placeholder
            self.network_weights["weights_gener"]["out"]._assign_op = assign_op
        feed_dict[assign_placeholder] = new_weights
        self.sess.run(assign_op, feed_dict=feed_dict)

    def topic_prop(self, X):
        """theta_ is the topic proportion vector. Apply softmax transformation to it before use.
        """
        # theta_ = self.sess.run((self.z),feed_dict={self.x: np.expand_dims(X, axis=0),self.keep_prob: 1.0})
        theta_ = self.sess.run((self.z), feed_dict={self.x: X, self.keep_prob: 1.0})
        return theta_
