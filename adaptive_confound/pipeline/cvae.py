from .. import utils as acu
from .. import control as acc
from .. import confound_detection as accd
from keras.backend.tensorflow_backend import set_session
from .data_pairs import DatasetPairs
import luigi
import json
import os.path
import pickle
import tensorflow as tf
import itertools as it

class RunCVAE(luigi.WrapperTask):
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
    
        params = list(it.product(tr_biases, te_biases, range(self.n_folds)))
        dependencies = []
        for i, (tr_bias, te_bias, foldidx) in enumerate(params):
            dependencies.append(
                CVAE_Task(
                    d = self.d,
                    tr_bias = tr_bias,
                    te_bias = te_bias,
                    foldidx = foldidx,
                    outdir = self.outdir,
                    max_gpu = self.max_gpu,
                    cvae_params = json.dumps(cvae_params),
                    cvae_fit_params = json.dumps(cvae_fit_params),
                    tr_frac = self.tr_frac,
                    random_seed = self.random_seed,
                    size = self.size
                )
            )
        return dependencies
        
class CVAE_Task(luigi.Task):    
    d = luigi.Parameter()
    tr_bias = luigi.FloatParameter()
    te_bias = luigi.FloatParameter()
    foldidx = luigi.IntParameter()
    outdir = luigi.Parameter()
    max_gpu = luigi.FloatParameter()
    cvae_params = luigi.Parameter()
    cvae_fit_params = luigi.Parameter()
    tr_frac = luigi.FloatParameter(default=.5)
    random_seed = luigi.IntParameter(default=1234)
    size = luigi.IntParameter(default=1000)
    
    def requires(self):
        return DatasetPairs(
            d=self.d,
            tr_bias=self.tr_bias,
            te_bias=self.te_bias,
            foldidx=self.foldidx,
            outdir=self.outdir,
            tr_frac=self.tr_frac,
            random_seed=self.random_seed,
            size=self.size,
        )
    
    def run(self):
        tr_path, te_path = self.input()
        with tr_path.open("r") as fd:
            d_tr = pickle.load(fd)
        with te_path.open("r") as fd:
            d_te = pickle.load(fd)
            
        # access luigi config to get the number of workers
        config = luigi.configuration.get_config()
        core = luigi.interface.core(config)
        self.max_gpu_per_task = self.max_gpu / core.workers
        
        # share the max GPU usage amongst workers
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.max_gpu_per_task
        sess = tf.Session(config=config)
        set_session(sess)
        
        # fit CVAE
        cvae_params = json.loads(self.cvae_params)
        cvae_fit_params = json.loads(self.cvae_fit_params)
        cvae = accd.ConditionalVAE(d_tr.X.shape[1], **cvae_params)
        cvae.fit(d_tr, d_te, **cvae_fit_params)
 
        with self.output()[0].open("w") as fd:
            fd.write(cvae.vae.to_json())
        with self.output()[1].open("w") as fd:
            fd.write(pickle.dumps(cvae.vae.get_weights()))

    def output(self):
        fname = "trbias={:.3f}_tebias={:.3f}_size={}_foldidx={}_trfrac={:.3f}".format(
            self.tr_bias, self.te_bias, self.size, self.foldidx, self.tr_frac
        )
        fnames = [ 
            os.path.join(self.outdir, "cvae_models", fname + _) 
            for _ in ["_arch.json", "_weights.pkl"] 
        ]
        return [
            luigi.LocalTarget(fnames[0]),
            luigi.LocalTarget(fnames[1], format=luigi.format.Nop)
        ]
