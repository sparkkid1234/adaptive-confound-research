import luigi
import json
import itertools as it
from ..utils.dataset import read_pickle
from .topic_distribution import *
from .ba_vs_lr import *
from .pxgz import HellingerDistance

# luigi --module adaptive_confound.pipeline MainTask --d /data/virgile/confound/adaptive/in/twitter_dataset_y=location_z=gender.pkl --outdir /data/virgile/confound/adaptive/luigi/twitter --workers 4 --max-gpu .5
class MainTask(luigi.WrapperTask):
    d = luigi.Parameter()
    max_gpu = luigi.FloatParameter()
    n_folds = luigi.IntParameter(default=3)
    n_topics = luigi.IntParameter(default=100)
    outdir = luigi.Parameter(default="/data/virgile/confound/adaptive/luigi/")
    tr_frac = luigi.FloatParameter(default=.5)
    random_seed = luigi.IntParameter(default=1234)
    size = luigi.IntParameter(default=1000)
    epochs = luigi.IntParameter(default=200)

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
        tm_tasks = ["TD_DNN", "TD_VAE", "TD_sProdLDA", "TD_LDA"]

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
            for tr_bias, te_bias, foldidx in params:
                #dependencies.append(
                #    HellingerDistance(
                #        d=self.d,
                #        tr_bias=tr_bias,
                #        te_bias=te_bias,
                #        foldidx=foldidx,
                #        n_topics=self.n_topics,
                #        outdir=self.outdir,
                #        max_gpu=self.max_gpu,
                #        tr_frac=self.tr_frac,
                #        random_seed=self.random_seed,
                #        size=self.size,
                #        model_params=json.dumps(tm_mp),
                #        log_params=json.dumps(tm_lp),
                #        fit_params=json.dumps(tm_fp),
                #        topic_distrib_task=tm_task
                #    )
                #)
                dependencies.append(
                    BAvsLR(
                        d=self.d,
                        tr_bias=tr_bias,
                        te_bias=te_bias,
                        foldidx=foldidx,
                        n_topics=self.n_topics,
                        outdir=self.outdir,
                        max_gpu=self.max_gpu,
                        tr_frac=self.tr_frac,
                        random_seed=self.random_seed,
                        size=self.size,
                        model_params=json.dumps(tm_mp),
                        log_params=json.dumps(tm_lp),
                        fit_params=json.dumps(tm_fp),
                        topic_distrib_task=tm_task,
                        ba_params=json.dumps(ba_params),
                        lr_params=json.dumps(lr_params),
                    )
                )
        return dependencies
