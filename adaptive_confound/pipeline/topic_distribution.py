import luigi
import os.path
import pickle
import json
import tensorflow as tf
import itertools as it
from abc import ABC, abstractmethod
from ..topic_model import *
from ..utils.dataset import concat, read_pickle
from .data_pairs import *
from keras.backend.tensorflow_backend import set_session
import os

GPU_COUNT = 4

class RunTopicDistribution(luigi.WrapperTask):
    d = luigi.Parameter()
    max_gpu = luigi.FloatParameter()
    n_folds = luigi.IntParameter(default=3)
    n_topics = luigi.IntParameter(default=100)
    outdir = luigi.Parameter(default="/data/virgile/confound/adaptive/luigi/")
    tr_frac = luigi.FloatParameter(default=.5)
    random_seed = luigi.IntParameter(default=1234)
    size = luigi.IntParameter(default=1000)
    epochs = luigi.IntParameter(default=300)

    def requires(self):
        ## Define topic models to use

        # parameters specific to prodLDA
        d = read_pickle(self.d)
        prodlda_network_architecture = dict(
            n_hidden_recog_1=200,  # 1st layer encoder neurons
            n_hidden_recog_2=200,  # 2nd layer encoder neurons
            n_hidden_recog_3=200,
            n_hidden_gener_1=d.X.shape[1],  # 1st layer decoder neurons
            n_input=d.X.shape[1],
            n_output=1,
            n_z=self.n_topics,
        )

        # list of topic model tasks
        tm_tasks = [TD_DNN, TD_VAE, TD_sProdLDA, TD_LDA]

        # list of associated model parameters
        tm_model_params = [
            dict(),
            dict(hidden_dim=self.n_topics),
            dict(
                network_architecture=prodlda_network_architecture,
                batch_size=32,
                learning_rate=0.01,
                loss="mse",
                trade_off=1,
                #trade_off can be set between 0 and 1 to favor discriminative loss
            ),
            dict(batch_size=32, random_state=self.random_seed)
        ]

        # list of associated log parameters
        tm_log_params = [dict(), dict(), dict(), dict()]

        # list of associated parameters fed at fitting time
        tm_fit_params = [
            dict(epochs=self.epochs, verbose=0),
            dict(epochs=self.epochs, verbose=0),
            dict(training_epochs=self.epochs, display_step=self.epochs // 10),
            dict()
        ]

        # list of biases at training and testing times
        # keep the following two for complete umbrella plot
        #tr_biases = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        #te_biases = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
        # keep the following two for simple no shift / low shift / high shift comparison
        tr_biases = [.1, .5, .9]
        te_biases = [.1, .5, .9]
        params = list(it.product(tr_biases, te_biases, range(self.n_folds)))

        # Parameters for BA and LR
        ba_params = dict()
        lr_params = dict()

        # create and return the list of required tasks
        dependencies = []
        for tm_task, tm_mp, tm_lp, tm_fp in zip(
            tm_tasks, tm_model_params, tm_log_params, tm_fit_params
        ):
            for i, (tr_bias, te_bias, foldidx) in enumerate(params):
                dependencies.append(
                    tm_task(
                        d=self.d,
                        tr_bias=tr_bias,
                        te_bias=te_bias,
                        foldidx=foldidx,
                        outdir=self.outdir,
                        tr_frac=self.tr_frac,
                        random_seed=self.random_seed,
                        size=self.size,
                        n_topics=self.n_topics,
                        max_gpu=self.max_gpu,
                        model_params=json.dumps(tm_mp),
                        log_params=json.dumps(tm_lp),
                        fit_params=json.dumps(tm_fp),
                        run_on_gpunum=i % GPU_COUNT
                    )
                )
        return dependencies

class TopicDistribution(luigi.Task, ABC):
    d = luigi.Parameter()
    tr_bias = luigi.FloatParameter()
    te_bias = luigi.FloatParameter()
    foldidx = luigi.IntParameter()
    outdir = luigi.Parameter()
    tr_frac = luigi.FloatParameter(default=.5)
    random_seed = luigi.IntParameter(default=1234)
    size = luigi.IntParameter(default=1000)
    
    n_topics = luigi.IntParameter()
    max_gpu = luigi.FloatParameter()
    model_params = luigi.Parameter(default="{}")
    log_params = luigi.Parameter(default="{}")
    fit_params = luigi.Parameter(default="{}")
    fit_on_both = luigi.BoolParameter(default=False)
    run_on_gpunum = luigi.IntParameter(default=0)
    
    retry_count = 3
    
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
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.run_on_gpunum % GPU_COUNT)
        tr_path, te_path = self.input()
        with tr_path.open("r") as fd:
            d_tr = pickle.load(fd)
        with te_path.open("r") as fd:
            d_te = pickle.load(fd)

        config = luigi.configuration.get_config()
        core = luigi.interface.core(config)
        self.max_gpu_per_task = self.max_gpu / core.workers
        self.tm = self.init_tm()
        topics = self.fit_transform_tm(d_tr, d_te)

        for topic_res, outpath in zip(topics, self.output()):
            with outpath.open("w") as fd:
                pickle.dump(topic_res, fd)

    def output(self):
        fname = "trbias={:.3f}_tebias={:.3f}_size={}_foldidx={}_trfrac={:.3f}".format(
            self.tr_bias, self.te_bias, self.size, self.foldidx, self.tr_frac
        )
        fpath = os.path.join(self.outdir, "topic_distrib", self.get_name(), fname)
        fpaths = [_.format(fpath) for _ in ["{}_train.pkl", "{}_test.pkl"]]
        return [luigi.LocalTarget(_, format=luigi.format.Nop) for _ in fpaths]

    def fit_transform_tm(self, d_tr, d_te):
        fit_params = json.loads(self.fit_params)
        if self.fit_on_both:
            d_fit = concat([d_tr, d_te])
        else:
            d_fit = d_tr
        self.tm.fit(d_fit, **fit_params)
        topics_tr = self.tm.transform(d_tr)
        topics_te = self.tm.transform(d_te)

        return topics_tr, topics_te

    def get_name(self):
        return self.__class__.__qualname__

    @abstractmethod
    def init_tm(self):
        """
        Initializes the topic model and returns it. The variables
        `self.model_params` are json strings that can be read using `json.loads`
        to initialize the model with user-defined parameters.
        """
        pass


class TD_KerasModel(TopicDistribution):
    @abstractmethod
    def get_topic_model_class(self):
        pass

    def init_tm(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.max_gpu_per_task
        sess = tf.Session(config=config)
        set_session(sess)

        model_params = json.loads(self.model_params)
        log_params = json.loads(self.log_params)
        d = read_pickle(self.d)

        tm = self.get_topic_model_class()(
            d.X.shape[1], self.n_topics, model_params, log_params
        )
        return tm


class TD_VAE(TD_KerasModel):
    def get_topic_model_class(self):
        return VariationalAutoEncoder

    def get_name(self):
        model_params = json.loads(self.model_params)
        supervised = model_params.get("supervised", False)
        return "TD_{}VAE".format("s" if supervised else "")


class TD_DNN(TD_KerasModel):
    def get_topic_model_class(self):
        return DenseNeuralNet

class TD_FromClass(TopicDistribution):
    def init_tm(self, cl):
        model_params = json.loads(self.model_params)
        if cl == DiscriminativeProdLDA:
            model_params["gpu_fraction"] = self.max_gpu_per_task
        log_params = json.loads(self.log_params)
        d = read_pickle(self.d)
        tm = cl(
            d.X.shape[1], self.n_topics, model_params, log_params
        )
        return tm
        
class TD_sProdLDA(TD_FromClass):
    def init_tm(self):
        return super().init_tm(DiscriminativeProdLDA)

class TD_LDA(TD_FromClass):
    def init_tm(self):
        return super().init_tm(LatentDirichletAllocation)