from keras.layers import *
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import *
from .. import utils as acu
from keras.utils import to_categorical
from keras.regularizers import Regularizer
from keras import losses
from keras import metrics
import keras.backend as K
import scipy.sparse as sp
import numpy as np
import itertools as it
import tensorflow as tf

# Reverse gradient layer from https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py
# - Added compute_output_shape for Keras 2 compatibility
# - Fixed bug where RegisterGradient was raising a KeyError
def reverse_gradient(X, hp_lambda):
    """Flips the sign of the incoming gradient during training."""
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    while True:
        try:
            grad_name = "GradientReversal%d" % reverse_gradient.num_calls

            @tf.RegisterGradient(grad_name)
            def _flip_gradients(op, grad):
                return [tf.negative(grad) * hp_lambda]

            break
        except KeyError:
            reverse_gradient.num_calls += 1

    g = K.get_session().graph
    with g.gradient_override_map({"Identity": grad_name}):
        y = tf.identity(X)
    return y


class GradientReversal(Layer):
    """Flip the sign of gradient during training."""

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"hp_lambda": self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def correlation_regularizer(w, scaler=1):
    """
    Activity regularizer that computes the absolute correlation between each pair
    of units across a batch then sum them.
    """
    fs = K.transpose( (w - K.mean(w, axis=0)) / K.std(w, axis=0) )
    N = K.int_shape(w)[1]
    xy_pairs = it.combinations(range(N), 2)
    
    to_gather_x, to_gather_y = zip(*xy_pairs)
    fsx, fsy = K.gather(fs, to_gather_x), K.gather(fs, to_gather_y)
    loss_xy = K.abs( K.mean(fsx * fsy, axis=0) )
    loss = scaler * K.mean(loss_xy)
    return loss

class CorrelationRegularizer(Layer):
    def __init__(self, l=1., **kwargs):
        super(CorrelationRegularizer, self).__init__(**kwargs)
        self.l = l
        self.activity_regularizer = lambda w: correlation_regularizer(w, scaler=l)
        
    def get_config(self):
        config = {'l': self.l}
        base_config = super(CorrelationRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    
class KerasProdLDA:
    def __init__(self, k, m, hidden_dim, batch_size, alpha, supervised=False,
                 y_inv_factor=-1, s_inv_factor=-1, beta0=1, beta1=1, beta2=1, beta3=1, beta4=1):
        self.k = k
        self.m = m
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.alpha = alpha
        self.supervised = supervised
        self.beta0 = beta0 # knob for the reconstruction loss (cross entropy)
        self.beta1 = beta1 # knob for the encoder loss (LDA)
        self.beta2 = beta2 if supervised else 0 # knob for the discriminative loss
        self.beta3 = beta3 # knob for the domain loss
        self.beta4 = beta4 # knob for the correlation regularization loss
        self.y_inv_factor = y_inv_factor
        self.s_inv_factor = s_inv_factor
        
        self.__build_model()
        
    def __build_model(self):
        K.clear_session()
        mu1 = np.log(self.alpha) - 1 / self.m * self.m * np.log(self.alpha) # 0
        sigma1 = 1 / self.alpha * (1 - 2 / self.m) + 1 / (self.m ** 2) * self.m / self.alpha
        inv_sigma1 = 1 / sigma1
        log_det_sigma = self.m * np.log(sigma1)

        x = Input(batch_shape=(self.batch_size, self.k))
        softmax_x = Lambda(lambda x: K.softmax(x))(x)

        h = Dense(self.hidden_dim, activation='softplus', name="H1")(x)
        h = Dense(self.hidden_dim, activation='softplus', name="H2")(h)
        
        corr_reg = lambda w: correlation_regularizer(w, scaler=self.beta4)
        z_mean = Dense(self.m,
                       name="z_mean")(h)
        z_mean = BatchNormalization()(z_mean)
        z_log_var = Dense(self.m,
                          name="z_log_var")(h)
        z_log_var = BatchNormalization()(z_log_var)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(self.batch_size, self.m), mean=0., stddev=1.)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        unnormalized_z = Lambda(sampling,
                                output_shape=(self.m,))([z_mean, z_log_var])

        theta = Activation('softmax', name="Z")(unnormalized_z)
        if self.beta4:
            theta = CorrelationRegularizer(l=self.beta4 * self.k)(theta)
        theta = Dropout(0.5)(theta)

        xhat = Dense(units=self.k, name="X_hat")(theta)
        xhat = BatchNormalization()(xhat)
        xhat = Activation("softmax", name="X")(xhat)

        next_layer_y = theta
        if self.y_inv_factor > 0:
            next_layer_y = GradientReversal(self.y_inv_factor)(theta)
        yhat = Dense(units=2, name="y_hat")(next_layer_y)
        yhat = BatchNormalization()(yhat)
        yhat = Activation("softmax", name="y")(yhat)
        
        next_layer_s = theta
        if self.s_inv_factor > 0:
            next_layer_s = GradientReversal(self.s_inv_factor)(theta)
        shat = Dense(units=2, name="s_hat")(next_layer_s)
        shat = BatchNormalization()(shat)
        shat = Activation("softmax", name="s")(shat)

        def prodlda_loss(x, xhat):
            decoder_loss = K.mean(x * K.log(xhat), axis=-1)
            decoder_loss *= -1 * self.k
            
            encoder_loss = inv_sigma1 * K.exp(z_log_var) + K.square(z_mean) * inv_sigma1 - 1 - z_log_var
            encoder_loss = K.sum(encoder_loss, axis=-1) + log_det_sigma
            encoder_loss *= .5
            return K.mean(self.beta0 * decoder_loss +
                          self.beta1 * encoder_loss)
        
        def discrim_loss(y, yhat):
            loss = categorical_crossentropy(y, yhat)
            loss *= self.k
            return K.mean(self.beta2 * loss)
        
        def domain_loss(s, shat):
            loss = categorical_crossentropy(s, shat)
            loss *= self.k
            return K.mean(self.beta3 * loss)

        self.model = Model(x, [xhat, yhat, shat])
        self.model.compile(optimizer=Adam(lr=0.001),#, beta_1=0.99),#RMSprop(lr=0.05),
                           loss={"X": prodlda_loss,
                                 "y": discrim_loss,
                                 "s": domain_loss})

        self.topic_model = Model(x, theta)
        self.label_clf = Model(x, yhat)

    def fill_matrix(self, X, y=None):
        n, m = X.shape
        remain = n % self.batch_size
        fill = 0
        new_X = X.copy()
        if remain:
            fill = self.batch_size - remain
            X_issparse = sp.issparse(X)
            stack_fn = sp.vstack if X_issparse else np.vstack
            empty_rows = sp.csr_matrix((fill, m)) if X_issparse else np.zeros((fill, m))
            new_X = stack_fn([X, empty_rows])

        if y is not None:
            if y.ndim == 1:
                new_y = np.hstack([y, np.zeros(fill)])
            else:
                new_y = np.vstack([y, np.zeros((fill, y.shape[1]))])
            return new_X, new_y, fill
        return new_X, fill

    def fit(self, d_tr, d_te=None, *args, **kwargs):
        y_tr = to_categorical(d_tr.y, 2) if self.supervised else np.zeros((d_tr.y.size,2))
        
        if d_te is not None:
            y_te = np.zeros((d_te.y.size, 2))
            d_all = acu.dataset.concat([d_tr, d_te])
            d_all.y = np.vstack([y_tr, y_te])
        else:
            d_all = d_tr
            
        filled_X, filled_y, fill_val = self.fill_matrix(d_all.X, d_all.y)
        to_stack = [
            np.array([[0,1] for _ in d_tr.y]),
            np.array([[1,0] for _ in d_te.y])
        ]
        if fill_val:
            to_stack.append(np.array([[0,0] for _ in range(fill_val)]))
        
        s_all = np.vstack(to_stack)
        return self.model.fit(filled_X,
                              [filled_X, filled_y, s_all],
                              *args, **kwargs)

    def transform(self, d, *args, **kwargs):
        filled_X, fill_val = self.fill_matrix(d.X)
        topics = self.topic_model.predict(filled_X, *args, **kwargs)
        if fill_val:
            topics = topics[:-fill_val]
        return topics
    
    def predict(self, d, *args, **kwargs):
        filled_X, fill_val = self.fill_matrix(d.X)
        ypred = self.label_clf.predict(filled_X, *args, **kwargs)
        if fill_val:
            ypred = ypred[:-fill_val]
        return ypred