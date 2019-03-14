from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.engine import Layer
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, History
from keras.optimizers import *
from glob import glob
from tqdm import tqdm
from scipy.stats import gmean
import numpy as np
import tensorflow as tf
import keras.backend as K
import os.path
import scipy.sparse as sp

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

class A_BOW:
    """
    Implements the Adversarial Selector with Bag of Words model (A+BOW)
    from Deconfounded Lexicon Induction for Interpretable Social Science
    using Keras.
    """

    def __init__(
        self,
        x_dim,
        z_dim,
        h1=100,
        h2=50,
        z_inv_factor=1,
        d_inv_factor=1,
        loss_weights=None,
        use_domain=False,
        checkpoint_dir=None,
        dropout_rate=.1,
        optimizer=None,
        z_loss="binary_crossentropy",
        z_activation="sigmoid",
        select_best=False
    ):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h1, self.h2 = h1, h2
        self.d_inv_factor = d_inv_factor
        self.z_inv_factor = z_inv_factor
        self.loss_weights = loss_weights
        self.use_domain = use_domain
        self.checkpoint_dir = "/tmp" if checkpoint_dir is None else checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.dropout_rate = dropout_rate
        self.cpt_name = os.path.join(self.checkpoint_dir, "abow.{epoch:05d}.hdf5")
        self.mcp = ModelCheckpoint(self.cpt_name, save_weights_only=True)
        self.history = History()
        self.optimizer = optimizer
        self.z_loss = z_loss
        self.z_activation = z_activation
        self.select_best = select_best
    
    def _build_model(self):
        x_input = Input((self.x_dim,), name="x_input")
        e = Dropout(self.dropout_rate)(x_input)
        e = Dense(self.h1, activation="relu", kernel_regularizer="l2", name="e")(e)

        # Predict y
        l = Dense(self.h2, activation="relu", kernel_regularizer="l2")(e)
        y = Dense(1, activation="sigmoid", name="y")(l)

        # Predict z + gradient reversal
        l = e
        if self.z_inv_factor > 0:
            l = GradientReversal(self.z_inv_factor)(l)
        l = Dense(self.h2, activation="relu", kernel_regularizer="l2")(l)
        z = Dense(self.z_dim, name="z", activation=self.z_activation)(l)

        # Label Predictor
        self.label_clf = Model(x_input, y)
        
        # Full Model
        outputs = [y,z]
        losses = dict(
            y = "binary_crossentropy",
            z = self.z_loss
        )
        
        if self.use_domain:
            l = e
            if self.d_inv_factor > 0:
                l = GradientReversal(self.d_inv_factor)(l)
            l = Dense(self.h2, activation="relu", kernel_regularizer="l2")(l)
            d = Dense(1, name="d", activation="sigmoid")(l)
            
            outputs.append(d)
            losses["d"] = "binary_crossentropy"
            
        self.model = Model(x_input, outputs)
        self.model.compile(
            optimizer=self.optimizer if self.optimizer is not None else Adam(lr=0.0001, beta_1=0.99),
            loss=losses,
            loss_weights=self.loss_weights,
            metrics=["accuracy"]
        )
        self.pred_model = Model(x_input, y)

    def fit(self, d, *args, **kwargs):
        K.clear_session()
        self._build_model()
        
        if not self.use_domain and not hasattr(d, "domain"):
            d.domain = np.zeros_like(d.y)
            
        # process validation data if it exists
        vd = kwargs.get("validation_data", ())
        if type(vd) != tuple:
            kwargs["validation_data"] = (vd.X, [vd.y, vd.z])

        # fit with checkpointing
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.mcp, self.history]

        inputs = d.X
        outputs = [d.y, d.z]
        if self.use_domain:
            outputs.append(d.domain)
        self.h = self.model.fit(inputs, outputs, *args, **kwargs)
        if self.select_best:
            self.load_best_model()
            
#    def load_best_model(self):
#        yloss = np.array(self.h.history["y_loss"])
#        zloss = np.array(self.h.history["z_loss"])
#        vals = yloss - zloss
#        if self.use_domain:
#            vals -= np.array(self.h.history["d_loss"])
#        best_epoch = vals.argmin() + 1
#        self.model.load_weights(self.cpt_name.format(epoch=best_epoch))

    def load_best_model(self):
        vals = [
            np.array(self.h.history["y_loss"]),
            - np.array(self.h.history["z_loss"])
        ]
        if self.use_domain:
            vals.append(- np.array(self.h.history["d_loss"]))
        
        vals = np.vstack(vals).sum(axis=0)
        best_epoch = np.argmin(vals) + 1
        self.model.load_weights(self.cpt_name.format(epoch=best_epoch))

    def predict(self, d, *args, **kwargs):
        preds = self.model.predict(d.X)[0].round().flatten()
        return preds