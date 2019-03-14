import luigi
import os.path
import numpy as np
import pickle
import json
import pandas as pd
from ..utils import *
from ..confound_detection import *
from .data_pairs import DatasetPairs
from . import topic_distribution as td


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / np.sum(e_x, axis=1).reshape(-1, 1)

class HellingerDistance(luigi.Task):
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

    # one of TD_DNN, TD_VAE, TD_sProdLDA, TD_LDA
    topic_distrib_task = luigi.Parameter()
    
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
            getattr(td, self.topic_distrib_task)(
                d=self.d,
                tr_bias=self.tr_bias,
                te_bias=self.te_bias,
                foldidx=self.foldidx,
                outdir=self.outdir,
                tr_frac=self.tr_frac,
                random_seed=self.random_seed,
                size=self.size,
                n_topics=self.n_topics,
                max_gpu=self.max_gpu,
                model_params=self.model_params,
                log_params=self.log_params,
                fit_params=self.fit_params,
                fit_on_both=self.fit_on_both,
            ),
        ]

    def run(self):
        (tr_path, te_path), (tr_topic_path, te_topic_path) = self.input()

        # load required data
        req_data = []
        for path in [tr_path, te_path, tr_topic_path, te_topic_path]:
            with path.open("r") as fd:
                req_data.append(pickle.load(fd))
        d_tr, d_te, topic_tr, topic_te = req_data
        
        pxgz_delta = np.zeros(self.n_topics)
        topic_tr, topic_te = [_.argmax(axis=1) for _ in [topic_tr, topic_te]]
        
        for zi in range(self.n_topics):
            topic_tri = (topic_tr == zi).astype(int)
            topic_tei = (topic_te == zi).astype(int)

            if not np.any(topic_tri) and not(np.any(topic_tei)):
                pxgz_delta[zi] = np.nan
            else:
                pxgz_delta[zi] = pxgz_diff_hd(
                    d_tr.X, d_tr.y, topic_tri,
                    d_te.X, d_tr.y, topic_tei,
                    2
                )
            
        r = {}
        r["pxgz_delta"] = pxgz_delta.tolist()
        r["corr_tr"] = d_tr.pearsonr[0]
        r["corr_te"] = d_te.pearsonr[0]
        r["bias_tr"] = d_tr.get_bias()
        r["bias_te"] = d_te.get_bias()
        
        with self.output().open("w") as fd:
            fd.write(json.dumps(r))
        
    def output(self):
        fname = "trbias={:.3f}_tebias={:.3f}_size={}_foldidx={}_trfrac={:.3f}.json".format(
            self.tr_bias, self.te_bias, self.size, self.foldidx, self.tr_frac
        )
        fpath = os.path.join(
            self.outdir, "pxgz_diff", self.topic_distrib_task, fname
        )
        return luigi.LocalTarget(fpath)