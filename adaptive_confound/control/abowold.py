import keras.backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.engine import Layer
import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint,History
from glob import glob
import numpy as np
from keras.backend.tensorflow_backend import set_session
import os.path
from tqdm import tqdm_notebook as tqdm
# Reverse gradient layer from https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py
# Added compute_output_shape for Keras 2 compatibility
def reverse_gradient(X, hp_lambda):
    """Flips the sign of the incoming gradient during training."""
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]
    reverse_gradient.num_calls += 1
    g = K.get_session().graph
    with g.gradient_override_map({"Identity": grad_name}):
        y = tf.identity(X)
    print(grad_name)
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


class A_BOW_OLD:
    """
    Implements the Adversarial Selector with Bag of Words model (A+BOW)
    from Deconfounded Lexicon Induction for Interpretable Social Science
    using Keras.
    """

    def __init__(
        self,
        x_dim,
        hx=100,
        ht=50,
        hc=50,
        inv_factor=1,
        use_ensemble_model=False,
        checkpoint_dir=None,
        p=.1,
        n=5,
    ):
        assert n > 1
        assert p > 0 and p <= 1
        self.x_dim = x_dim
        self.hx, self.ht, self.hc = hx, ht, hc
        self.inv_factor = inv_factor
        self.checkpoint_dir = "/tmp" if checkpoint_dir is None else checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.use_ensemble_model = use_ensemble_model
        self.p = p
        self.n = n
        if self.n % 2 == 0:
            self.n -= 1
        self.cpt_name = os.path.join(self.checkpoint_dir, "abow.{epoch:05d}.hdf5")
        self.mcp = ModelCheckpoint(self.cpt_name, save_weights_only=True)
        self.history = History()
        self.model_paths = None

    def _build_model(self):
        K.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.125
        sess = tf.Session(config=config)
        set_session(sess)
        x_input = Input((self.x_dim,), name="x_input")
        e = Dense(self.hx, activation="relu", name="e")(x_input)

        l = Dense(self.ht, activation="relu")(e)
        t = Dense(2, activation="softmax", name="y")(l)

        l = GradientReversal(self.inv_factor)(e)
        l = Dense(self.hc, activation="relu")(l)
        c = Dense(2, activation="softmax", name="z")(l)
        
        self.model = Model(x_input, [t, c])
        self.model.compile(optimizer="adam", loss=["categorical_crossentropy","categorical_crossentropy"])

    def fit(self, d, *args, **kwargs):
        self._build_model()
        tocat = lambda x: to_categorical(x, num_classes=2)
        vd = kwargs.get("validation_data", ())
        if type(vd) != tuple:
            kwargs["validation_data"] = (vd.X, [tocat(vd.y), vd.z])
        
        # fit with checkpointing
        kwargs["callbacks"] = kwargs.get("callbacks", []) + [self.mcp, self.history]
        self.model.fit(d.X, [tocat(d.y), tocat(d.z)], *args, **kwargs)
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

    def predict(self, d, *args, **kwargs):
        return self.model.predict(d.X)[0]
#     def predict(self, d, *args, **kwargs):
#         get_predictions = lambda model: model.predict(d.X)[0].argmax(axis=1)
#         if self.model_paths is not None:
#             preds = []
#             for m in tqdm(self.model_paths):
#                 self.model.load_weights(m)
#                 preds.append(get_predictions(self.model))
#             votes = np.vstack(preds)
#             preds = np.median(votes, axis=0).astype(int)
#         else:
#             preds = get_predictions(self.model)
#         return preds
    
    def _select_n_best_models(self, d, yl, zl, model_paths):
        nmodels = len(model_paths)
        # keep models in the top 10% performance when predicting y
        k = int(self.p * nmodels)
        midx_list = yl.argsort()[:k]
        zidx_list = zl[midx_list].argsort()[::-1][: self.n]
        self.model_paths = model_paths[midx_list[zidx_list]]
        print("Using ensemble method with models {}".format(self.model_paths))

    def _select_one_best_model(self, d, yl, zl, model_paths):
        nmodels = len(model_paths)
        # keep models in the top 10% performance when predicting y
        k = int(self.p * nmodels)
        midx_list = yl.argsort()[:k]

        # keep the model that performs the worst on z
        zidx = zl[midx_list].argmax()
        midx = midx_list[zidx]
        print(
            "Reloading model {} with y loss {} and z loss {}".format(
                model_paths[midx], yl[midx], zl[midx]
            )
        )
        # load the selected model
        self.model.load_weights(model_paths[midx])