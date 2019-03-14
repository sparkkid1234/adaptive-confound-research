from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.engine import Layer
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, History
from glob import glob
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import keras.backend as K
import os.path
import scipy.sparse as sp

# Implements ver 1: c and c_hat is separate
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


class A_BOW_ADAPT:
    """
    Implements the Adversarial Selector with Bag of Words model (A+BOW)
    from Deconfounded Lexicon Induction for Interpretable Social Science
    using Keras.
    """

    def __init__(
        self,
        x_dim,
        z_dim,
        z_hat_dim,
        hx=100,
        ht=50,
        hc=50,
        inv_factor=1,
        yz_weight_ratio=1,
        use_last_epoch_model=True,
        use_ensemble_model=False,
        checkpoint_dir=None,
        dropout_rate=.1,
        optimizer="sgd",
        p=.1,
        n=5,
        z_loss="mean_absolute_error",
        z_hat_loss="mean_absolute_error",
        z_activation=None,
        z_hat_activation=None
    ):
        assert n > 1
        assert p > 0 and p <= 1
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.z_hat_dim = z_hat_dim
        self.hx, self.ht, self.hc = hx, ht, hc
        self.inv_factor = inv_factor
        self.yz_weight_ratio = yz_weight_ratio
        self.use_last_epoch_model = use_last_epoch_model
        self.checkpoint_dir = "/tmp" if checkpoint_dir is None else checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.dropout_rate = dropout_rate
        self.use_ensemble_model = use_ensemble_model
        self.p = p
        self.n = n
        if self.n % 2 == 0:
            self.n -= 1
        self.cpt_name = os.path.join(self.checkpoint_dir, "abow.{epoch:05d}.hdf5")
        self.mcp = ModelCheckpoint(self.cpt_name, save_weights_only=True)
        self.history = History()
        self.model_paths = None
        self.optimizer = optimizer
        self.z_loss = z_loss
        self.z_activation = z_activation
        self.z_hat_loss = z_hat_loss
        self.z_hat_activation = z_hat_activation

    def _build_model(self):
        x_input = Input((self.x_dim,), name="x_input")
        l = Dropout(self.dropout_rate)(x_input)
        e = Dense(self.hx, activation="relu", kernel_regularizer="l2", name="e")(l)

        l = Dense(self.ht, activation="relu", kernel_regularizer="l2")(e)
        t = Dense(1, activation="sigmoid", name="y")(l)

        l = GradientReversal(self.inv_factor)(e)
        l = Dense(self.hc, activation="relu", kernel_regularizer="l2")(l)
        c = Dense(self.z_dim, name="z", activation=self.z_activation)(l)
        
        l_1 = GradientReversal(self.inv_factor)(e)
        l_1 = Dense(self.hc, activation="relu",kernel_regularizer="l2")(l_1)
        c_hat = Dense(self.z_hat_dim, name="z_hats", activation=self.z_hat_activation)(l_1)
        
        
        # Domain Classifier
        self.domain_clf = Model(x_input, c)
        self.domain_clf.compile(optimizer=self.optimizer, loss=self.z_loss)
        #self.domain_clf = Model(x_input, [c, c_hat])
        #self.domain_clf.compile(optimizer=self.optimizer,loss=[self.z_loss,
        # Label Predictor
        self.label_clf = Model(x_input, t)
        self.label_clf.compile(optimizer=self.optimizer, loss="binary_crossentropy")
        
        # Full Model
        #<!!!!!!!!!> in this model c and c_hat is separated
        self.model = Model(x_input, [t, c, c_hat])
        self.model.compile(
            optimizer=self.optimizer,
            loss=["binary_crossentropy", self.z_loss, self.z_hat_loss],
            loss_weights=[self.yz_weight_ratio, 1,1]
        )

    def fit(self, d, *args, **kwargs):
        K.clear_session()
        self._build_model()

        # process validation data if it exists
        vd = kwargs.get("validation_data", ())
        if type(vd) != tuple:
            kwargs["validation_data"] = (vd.X, [vd.y, vd.z])

        # fit with checkpointing
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.mcp, self.history]

        self.model.fit(d.X, [d.y, d.z], *args, **kwargs)

        if not self.use_last_epoch_model:
            yl, zl = (
                np.array(self.history.history["y_loss"]),
                np.array(self.history.history["z_loss"]),
            )
            nmodels = len(yl)
            model_paths = np.array(
                [self.cpt_name.format(epoch=epoch) for epoch in range(1, nmodels + 1)]
            )
            if self.use_ensemble_model:
                self._select_n_best_models(d, yl, zl, model_paths)
            else:
                self._select_one_best_model(d, yl, zl, model_paths)

    def fit_domain_adaptation(self, d_tr, d_te, epochs,**kwargs):
        K.clear_session()
        self._build_model()
        yl=[]
        cl=[]
        zl=[]
        # Build domain label
        s_tr = np.repeat([[0,1]], d_tr.X.shape[0], axis=0)
        s_te = np.repeat([[1,0]], d_te.X.shape[0], axis=0)
        kwargs["callbacks"] = kwargs.get("callbacks",[])+[self.history]
        # Fit the full model first with training data, then fit the domain classifier with all data.
        # Repeat for each epoch
        for i in range(epochs):
            self.model.fit(d_tr.X, [d_tr.y, s_tr, d_tr.z],verbose=0,**kwargs)#, verbose=2)
            #<!!!!!!!> also fit domain_clf on d_te.z? No batch_size??
            self.domain_clf.fit(d_te.X, s_te,verbose=0)#, verbose=2)
            filepath = self.cpt_name.format(epoch=i+1)
            self.model.save_weights(filepath,overwrite=True)
            yl.extend(self.history.history["y_loss"])
            zl.extend(self.history.history["z_hats_loss"])
            cl.extend(self.history.history["z_loss"])
       
        if not self.use_last_epoch_model:
            yl, zl, cl = (
                np.array(yl),
                np.array(zl),
                np.array(cl)
            )
            nmodels = len(yl)
            self.model_paths = np.array(
                [self.cpt_name.format(epoch=epoch) for epoch in range(1, nmodels + 1)]
            )
#             if self.use_ensemble_model:
#                 self._select_n_best_models(d, yl, zl, model_paths)
#             else:
#                 self._select_one_best_model(yl, zl, cl, model_paths)
            
    def predict(self, d, *args, **kwargs):
        kwargs["mode"] = kwargs.get("mode","domain")
        get_predictions = lambda model: model.predict(d.X).round().flatten()
        if kwargs["mode"] == "domain":
            model = self.label_clf
        else:
            model = self.model
#         if self.model_paths is not None:
#             preds = []
#             for m in self.model_paths:
#                 self.model.load_weights(m)
#                 preds.append(get_predictions(model))
#             votes = np.vstack(preds)
#             preds = np.median(votes, axis=0).astype(int)
#         else:
        preds = get_predictions(model)
        return preds
    
    def _select_n_best_models(self, d, yl, zl, model_paths):
        nmodels = len(model_paths)
        # keep models in the top 10% performance when predicting y
        k = max(1, int(self.p * nmodels))
        midx_list = yl.argsort()[:k]
        zidx_list = zl[midx_list].argsort()[::-1][: self.n]
        self.model_paths = model_paths[midx_list[zidx_list]]
        print("Using ensemble method with models {}".format(self.model_paths))

    def _select_one_best_model(self, yl, zl, cl, model_paths):
        nmodels = len(model_paths)
        #zl_tmp = zl.copy()
        # keep models in the top 10% performance when predicting y
        k = int(self.p * nmodels)
        midx_list = yl.argsort()[:k]

        # keep the model that performs the worst on z
        zidx = zl[midx_list].argmax()
#         cidx = cl[midx_list].argmax()
        midx = midx_list[zidx]
        #cidx = cl[midx_list].argsort()[:7]
        #zidx = zl[midx_list].argsort()[:7]
        
        
        
        print(
            "Reloading model {} with y loss {}, z_loss {} and z_hat loss {}".format(
                model_paths[midx], yl[midx], cl[midx], zl[midx]
            )
        )
        # load the selected model
        self.model.load_weights(model_paths[midx])