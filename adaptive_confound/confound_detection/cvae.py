from keras import backend as K
from keras.layers import Dense, Input, Lambda, Concatenate, Dropout
from keras.models import Model
from keras.losses import binary_crossentropy, mse
from sklearn.linear_model import LogisticRegression
from .. import utils as acu
import scipy.sparse as sp
import numpy as np


class ConditionalVAE:
    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim,
        dropout_rate=.25,
        beta=1,
        regularization="l2",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.regularization = regularization
        self._build_model()

    def _sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def _build_model(self):
        ## ENCODER
        # build encoder model
        x_input = Input(shape=(self.input_dim,), name="input")
        s_input = Input(shape=(4,), name="src")

        x = Concatenate()([x_input, s_input])
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.hidden_dim, activation="relu")(x)

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        # use reparameterization trick to push the sampling out as input
        z = Lambda(self._sampling, name="z")([z_mean, z_log_var])
        z = Concatenate()([z, s_input])

        # instantiate encoder model
        self.encoder = Model([x_input, s_input], [z_mean, z_log_var, z])

        ## DECODER
        # build decoder model
        latent_units = Input(shape=(self.latent_dim + 4,))
        x = Dense(self.hidden_dim, activation="relu")(latent_units)
        outputs = Dense(self.input_dim, activation="sigmoid")(x)

        # instantiate decoder model
        self.decoder = Model(latent_units, outputs)

        ## TOPIC MODEL
        outputs = self.decoder(self.encoder([x_input, s_input])[2])

        def vae_loss(inp, out):
            repr_loss = self.input_dim * binary_crossentropy(x_input, outputs)
            kl_loss = (
                -0.5
                * self.beta
                * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            )
            return K.mean(repr_loss + kl_loss)

        self.vae = Model([x_input, s_input], outputs, name="vae_mlp")
        self.vae.compile(optimizer="adam", loss=vae_loss)

    def fit(self, d_train, d_test, **kwargs):
        s_tr = np.array([[0, 1] for _ in range(d_train.get_size())])
        s_te = np.array([[1, 0] for _ in range(d_test.get_size())])
        s = np.vstack([s_tr, s_te])

        y_tr = np.array([[0, 1] if _ else [1, 0] for _ in d_train.y])
        y_te = np.array([[0, 0] for _ in range(d_test.get_size())])
        y = np.vstack([y_tr, y_te])
        sy = np.hstack([s, y])

        d = acu.dataset.concat([d_train, d_test])
        self.vae.fit([d.X, sy], d.X, **kwargs)

    def transform(self, d, kind, **kwargs):
        """
        kind must be one of ["train", "test"]
        """
        n = d.X.shape[0]
        s = np.vstack([[0, 1] if kind == "train" else [1, 0] for _ in range(n)])
        y = np.vstack([[0, 1] if yi else [1, 0] for yi in d.y])
        sy = np.hstack([s, y])
        return self.vae.predict([d.X, sy])

    def get_train_test_diff(self, d):
        n = d.X.shape[0]
        dX2tr = self.transform(d, "train").sum(axis=0)
        dX2te = self.transform(d, "test").sum(axis=0)
        diff_trte = np.abs(dX2tr - dX2te)
        diff_trte_sparse = sp.csr_matrix(
            (diff_trte.reshape(1, -1).repeat(n, axis=0)[d.X.nonzero()], d.X.nonzero()),
            shape=d.X.shape,
        )
        return diff_trte_sparse

    def get_train_test_ratio(self, d):
        n, m = d.X.shape
        dX2tr = self.transform(d, "train").mean(axis=0)
        dX2te = self.transform(d, "test").mean(axis=0)
        ratio_trte = ( dX2tr + 1 ) / ( dX2te + 1 )
        ratio_trte_sparse = sp.csr_matrix(
            (ratio_trte.reshape(1, -1).repeat(n, axis=0)[d.X.nonzero()], d.X.nonzero()),
            shape=(n,m),
        )
        return ratio_trte_sparse
    
    def get_latent_repr(self, d, kind):
        n = d.X.shape[0]
        s = np.vstack([[0, 1] if kind == "train" else [1, 0] for _ in range(n)])
        y = np.vstack([[0, 1] if yi else [1, 0] for yi in d.y])
        sy = np.hstack([s, y])
        return self.encoder.predict([d.X, sy])