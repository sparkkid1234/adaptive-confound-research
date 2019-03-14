from ..utils import Dataset
from .interface import TopicModel
from keras.layers import Dense, Input, Lambda, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.metrics import mse, binary_crossentropy
from datetime import datetime
import keras.backend as K
import os.path
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / np.sum(e_x, axis=1).reshape(-1, 1)


class DenseNeuralNet(TopicModel):
    def __init__(self, k, m, model_params=None, log_params=None):
        if model_params is None:
            model_params = {}
        if log_params is None:
            log_params = {}
        self.apply_softmax = model_params.get("apply_softmax", True)
        super().__init__(k, m, model_params, log_params)

    def _build_model(self):
        """
        Create the Dense Neural Network topic model using Keras.        
        """
        # Define the layers
        inputs = Input(shape=(self.k,), name="bow_input")
        x = Dropout(.3)(inputs)
        topics = Dense(
            self.m, activation="relu", kernel_regularizer="l1", name="latent_variables"
        )(x)
        x = Dropout(.3)(topics)
        output = Dense(1, activation="sigmoid", name="pred_proba")(x)

        # Initialize the classifier we aim to optimize
        self.clf = Model(inputs, output, name="bow_clf")
        self.clf.compile(optimizer="adam", loss="binary_crossentropy")

        # Initialize the topic model (input -> topics)
        self.tm = Model(inputs, topics, name="topic_model")

    def partial_fit(self, d, *args, **kwargs):
        # Add TensorBoard callback
        tboard_cb = TensorBoard(**self.log_params)
        callbacks = kwargs.get("callbacks", [])
        callbacks.append(tboard_cb)
        kwargs["callbacks"] = callbacks

        # Fit the model
        self.clf.fit(d.X, d.y, *args, **kwargs)
        return self

    def fit(self, d, *args, **kwargs):
        # Reinitialize the model
        self._build_model()
        return self.partial_fit(d, *args, **kwargs)

    def transform(self, d, *args, **kwargs):
        r = self.tm.predict(d.X, *args, **kwargs)
        if self.apply_softmax:
            r = softmax(r)
        return r

    def get_name(self):
        if self._name is None:
            self._name = "DNN({},{},{})".format(self.k, self.m, 1)
        return self._name


class VariationalAutoEncoder(TopicModel):
    def __init__(self, k, m, model_params=None, log_params=None):
        """
        Initialize a VAE topic model.

        Args:
            k (int): The number of features expected by the model.
            m (int): The number of topics to generate.
            model_params (dict, optional): Dictionary used to pass configuration parameters to the model. The valid additional settings for VAE are:
               - hidden_dim (int): The size of the hidden dimension in VAE (default to 5% of the input dimension).
               - beta (int): The weight put on the KL loss (default to 1).
               - supervised (bool): Flag set to True to create a supervised VAE (default to False).
            log_params (dict, optional): Optional dictionary to store the logging information to use with TensorBoard.
        """
        if model_params is None:
            model_params = {}
        if log_params is None:
            log_params = {}
        # Retrieve additional settings for this model
        self.hidden_dim = model_params.get("hidden_dim", int(0.05 * k))
        self.beta = model_params.get("beta", 1)
        self.supervised = model_params.get("supervised", False)
        self.apply_softmax = model_params.get("apply_softmax", True)
        super().__init__(k, m, model_params, log_params)

    def _build_model(self):
        ## ENCODER
        # build encoder model
        inputs = Input(shape=(self.k,), name="encoder_input")
        x = Dropout(.2, seed=1234)(inputs)
        x = Dense(self.hidden_dim, activation="relu")(x)
        z_mean = Dense(self.m, name="z_mean")(x)
        z_log_var = Dense(self.m, name="z_log_var")(x)

        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean=0 and std=1.0
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        # use reparameterization trick to push the sampling out as input
        z = Lambda(sampling, name="z")([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

        ## DECODER
        # build decoder model
        latent_inputs = Input(shape=(self.m,), name="z_sampling")
        x = Dense(self.hidden_dim, activation="relu")(
            latent_inputs
        )
        x = Dropout(.2, seed=1234)(x)
        outputs = Dense(self.k, activation="sigmoid")(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name="decoder")

        def vae_loss(inp, out):
            repr_loss = self.k * mse(inp, out)
            kl_loss = (
                -0.5
                * self.beta
                * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            )
            return K.mean(repr_loss + kl_loss)

        ## TOPIC MODEL
        outputs = self.decoder(self.encoder(inputs)[2])
        self.topic_model = Model(inputs, outputs, name="vae_mlp")
        self.topic_model.compile(optimizer="adam", loss={"decoder": vae_loss})

        if self.supervised:
            ## CLASSIFIER
            # build predictive model from the latent representation
            y_output = Dense(1, activation="sigmoid")(latent_inputs)

            # instantiate predictive model
            classifier = Model(latent_inputs, y_output, name="classifier")
            predictions = classifier(self.encoder(inputs)[2])

            # instantiate full model
            self.full_model = Model(inputs, [outputs, predictions], name="full_model")
            self.full_model.compile(
                optimizer="adam",
                loss={"decoder": vae_loss, "classifier": "binary_crossentropy"},
            )
            self.classifier = classifier

        else:
            self.full_model = self.topic_model

    def fit(self, d, *args, **kwargs):
        # (Re)initialize the model
        self._build_model()

        # Add TensorBoard callback
        tboard_cb = TensorBoard(**self.log_params)
        callbacks = kwargs.get("callbacks", [])
        callbacks.append(tboard_cb)
        kwargs["callbacks"] = callbacks

        kwargs1 = {k: v for k, v in kwargs.items()}
        kwargs2 = {k: v for k, v in kwargs.items()}

        epochs = kwargs.get("epochs", 5)
        if not self.supervised:
            if type(epochs) != int:
                raise AttributeError(
                    "When using a supervised VAE, epochs argument can only be an integer."
                )
            else:
                kwargs1["epochs"] = epochs
        else:
            if type(epochs) == list:
                kwargs1["epochs"], kwargs2["epochs"] = epochs
            else:
                kwargs1["epochs"] = kwargs2["epochs"] = epochs

        # Compile the model to (re)initialize the weights and fit it
        self.topic_model.fit(d.X, d.X, *args, **kwargs1)
        if self.supervised:
            self.full_model.fit(d.X, [d.X, d.y], *args, **kwargs2)
        return self

    def transform(self, d, *args, **kwargs):
        enc_X = self.encoder.predict(d.X, *args, **kwargs)[2]
        if self.apply_softmax:
            enc_X = softmax(enc_X)
        return enc_X

    def get_name(self):
        if self._name is None:
            self._name = "{}VAE({},{},{})".format(
                "s" if self.supervised else "", self.k, self.hidden_dim, self.m
            )
        return self._name
