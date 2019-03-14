from .cvae import CVAE_Task
from .. import confound_detection as accd
from .. import utils as acu
from .. import control as acc
from keras.backend.tensorflow_backend import set_session
from sklearn.linear_model import LogisticRegression
from .data_pairs import DatasetPairs
import luigi
import json
import os.path
import pickle
import tensorflow as tf
import itertools as it
import pandas as pd
import os
import numpy as np

GPU_COUNT = 4

class RunCVAE_ABOW(luigi.WrapperTask):    
    d = luigi.Parameter()
    max_gpu = luigi.FloatParameter()
    n_folds = luigi.IntParameter(default=3)
    outdir = luigi.Parameter(default="/data/virgile/confound/adaptive/luigi/")
    tr_frac = luigi.FloatParameter(default=.5)
    random_seed = luigi.IntParameter(default=1234)
    size = luigi.IntParameter(default=1000)
    epochs = luigi.IntParameter(default=300)
    
    def requires(self):
        d = acu.read_pickle(self.d)
        
        tr_biases = [.1,.5,.9]
        te_biases = [.1,.5,.9]
        n_folds = 3
        
        cvae_params = dict(
            latent_dim = 200,
            hidden_dim = 300,
            dropout_rate = .1,
            beta = .1
        )
        cvae_fit_params = dict(epochs=self.epochs, verbose=2)
    
        abow_params = dict(
            hx=500,
            ht=200,
            hc=200,
            use_last_epoch_model=False,
            use_ensemble_model=True,
            n=9,
            p=.2,
            optimizer="adadelta",
            z_loss="mean_squared_error",
            yz_weight_ratio=1,
            inv_factor=1.5
        )
        abow_fit_params = dict(epochs=self.epochs, verbose=2)
        
        params = list(it.product(tr_biases, te_biases, range(self.n_folds)))
        dependencies = []
        for i, (tr_bias, te_bias, foldidx) in enumerate(params):
            abow_params["checkpoint_dir"] = "/data/virgile/checkpointing/{:03d}/".format(i)
            dependencies.append(
                CVAE_ABOW_Task(
                    d = self.d,
                    tr_bias = tr_bias,
                    te_bias = te_bias,
                    foldidx = foldidx,
                    outdir = self.outdir,
                    max_gpu = self.max_gpu,
                    cvae_params = json.dumps(cvae_params),
                    cvae_fit_params = json.dumps(cvae_fit_params),
                    abow_params = json.dumps(abow_params),
                    abow_fit_params = json.dumps(abow_fit_params),
                    tr_frac = self.tr_frac,
                    random_seed = self.random_seed,
                    size = self.size,
                    run_on_gpunum = i % GPU_COUNT
                )
            )
        return dependencies

class CVAE_ABOW_Task(luigi.Task):
    d = luigi.Parameter()
    tr_bias = luigi.FloatParameter()
    te_bias = luigi.FloatParameter()
    foldidx = luigi.IntParameter()
    outdir = luigi.Parameter()
    max_gpu = luigi.FloatParameter()
    cvae_params = luigi.Parameter()
    cvae_fit_params = luigi.Parameter()
    abow_params = luigi.Parameter()
    abow_fit_params = luigi.Parameter()
    tr_frac = luigi.FloatParameter(default=.5)
    random_seed = luigi.IntParameter(default=1234)
    size = luigi.IntParameter(default=1000)
    run_on_gpunum = luigi.IntParameter(default=0)
    
    def requires(self):
        return [
            DatasetPairs(
                d=self.d,
                tr_bias=self.tr_bias,
                te_bias=self.te_bias,
                foldidx=self.foldidx,
                outdir=self.outdir,
                tr_frac=self.tr_frac,
                random_seed=self.random_seed,
                size=self.size,
            ),
            CVAE_Task(
                d=self.d,
                tr_bias=self.tr_bias,
                te_bias=self.te_bias,
                foldidx=self.foldidx,
                outdir=self.outdir,
                max_gpu=self.max_gpu,
                cvae_params=self.cvae_params,
                cvae_fit_params=self.cvae_fit_params,
                tr_frac=self.tr_frac,
                random_seed=self.random_seed,
                size=self.size
            )
        ]
    
    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.run_on_gpunum % GPU_COUNT)
        
        # access luigi config to get the number of workers
        config = luigi.configuration.get_config()
        core = luigi.interface.core(config)
        self.max_gpu_per_task = self.max_gpu / core.workers
        
        # share the max GPU usage amongst workers
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.max_gpu_per_task
        sess = tf.Session(config=config)
        set_session(sess)
        
        # load input data
        (tr_path, te_path), (arch_path, weights_path) = self.input()
        with tr_path.open("r") as fd:
            d_tr = pickle.load(fd)
        with te_path.open("r") as fd:
            d_te = pickle.load(fd)
        with arch_path.open("r") as fd:
            arch = json.load(fd)
        with weights_path.open("r") as fd:
            weights = pickle.load(fd)
            
        layers = arch["config"]["layers"]
        input_dim = layers[0]["config"]["batch_input_shape"][-1]
        enc_arch = layers[2]["config"]["layers"]
        #dropout_rate = enc_arch[3]["config"]["rate"]
        hidden_dim = enc_arch[4]["config"]["units"]
        latent_dim = enc_arch[5]["config"]["units"]
        
        cvae = accd.ConditionalVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        cvae._build_model()
        cvae.vae.set_weights(weights)
        
        abow_results = []
        # fit LR baseline and predict
        lr = LogisticRegression()
        lr.fit(d_tr.X, d_tr.y)
        abow_results.append(dict(
            y_pred = lr.predict(d_te.X).tolist(),
            y_prob = lr.predict_proba(d_te.X).tolist(),
            y_true = d_te.y.tolist(),
            model = "LR",
            corr_tr = d_tr.pearsonr[0],
            corr_te = d_te.pearsonr[0],
            bias_tr = d_tr.get_bias(),
            bias_te = d_te.get_bias()
        ))
        
        # fit ABOW
        X_tr2te = cvae.transform(d_tr, "test")
        E_tr2tr = cvae.get_latent_repr(d_tr, "train")[2]
        E_tr2te = cvae.get_latent_repr(d_tr, "test")[2]
        ze = np.hstack([E_tr2tr[:, :-4], E_tr2te[:, :-4]])
        
        # compute diff between train and test, then fit A+BOW
        abow_params = json.loads(self.abow_params)
        abow = acc.A_BOW(d_tr.X.shape[1],
                         1 if len(ze.shape) == 1 else ze.shape[1],
                         **abow_params)
        d_cvae = acu.Dataset(
            X = X_tr2te,
            y = d_tr.y,
            z = ze #cvae.get_train_test_diff(d_tr)
        )
        abow_fit_params = json.loads(self.abow_fit_params)
        abow.fit(d_cvae, **abow_fit_params)
        abow_results.append(dict(
            y_pred = abow.predict(d_te).tolist(),
            y_true = d_te.y.tolist(),
            model = "CVAE+ABOW",
            corr_tr = d_tr.pearsonr[0],
            corr_te = d_te.pearsonr[0],
            bias_tr = d_tr.get_bias(),
            bias_te = d_te.get_bias()
        ))
        
        abow_results = pd.DataFrame(abow_results)
        with self.output().open("w") as fd:
            fd.write(abow_results.to_json(orient="records", lines=True))
            
    def output(self):
        fname = "trbias={:.3f}_tebias={:.3f}_size={}_foldidx={}_trfrac={:.3f}.jsonl".format(
            self.tr_bias, self.te_bias, self.size, self.foldidx, self.tr_frac
        )
        fname = os.path.join(self.outdir, "cvae_abow", fname) 
        return luigi.LocalTarget(fname)
