import luigi
import os.path
import numpy as np
import pickle
import json
import pandas as pd
from ..utils import *
from sklearn.linear_model import LogisticRegression
from ..control import BackdoorAdjustment
from .data_pairs import DatasetPairs
from . import topic_distribution as td
from .pxgz import HellingerDistance

class BAvsLR(luigi.Task):
    """
    """

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
    ba_params = luigi.Parameter(default="{}")
    lr_params = luigi.Parameter(default="{}")

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
            HellingerDistance(
                d=self.d,
                tr_bias=self.tr_bias,
                te_bias=self.te_bias,
                foldidx=self.foldidx,
                n_topics=self.n_topics,
                outdir=self.outdir,
                max_gpu=self.max_gpu,
                tr_frac=self.tr_frac,
                random_seed=self.random_seed,
                size=self.size,
                model_params=self.model_params,
                log_params=self.log_params,
                fit_params=self.fit_params,
                topic_distrib_task=self.topic_distrib_task
            )
        ]

    def run(self):
        (tr_path, te_path), (tr_topic_path, te_topic_path), topic_diffs_path = self.input()

        # load required data
        req_data = []
        for path in [tr_path, te_path, tr_topic_path, te_topic_path]:
            with path.open("r") as fd:
                req_data.append(pickle.load(fd))
        d_tr, d_te, topic_tr, topic_te = req_data
        int_topic_tr = topic_tr.argmax(axis=1)
        
        # load p(x|z) value and select z_i with the maximum value
        with topic_diffs_path.open("r") as fd:
            topic_diffs = json.load(fd)['pxgz_delta']
            topic_diffs = np.array([0 if np.isnan(_) else _ for _ in topic_diffs])
        c_idx, c_val = topic_diffs.argmax(), topic_diffs.max()
        
        lr_params = json.loads(self.lr_params)
        ba_params = json.loads(self.ba_params)
        lr = LogisticRegression(**lr_params)
        ba = BackdoorAdjustment(**ba_params)

        results = []

        # Compare to Logistic Regression
        lr.fit(d_tr.X, d_tr.y) 
        results.append(dict(
            y_prob=lr.predict_proba(d_te.X),
            y_pred=lr.predict(d_te.X),
            y_true=d_te.y,
            model="LR"
        ))

        # Compare to Backdoor Adjustment with true topic
        ba.fit(d_tr.X, d_tr.z.reshape(-1, 1), d_tr.y)
        results.append(dict(
            y_prob=ba.predict_proba(d_te.X),
            y_pred=ba.predict(d_te.X),
            y_true=d_te.y,
            model="BA"
        ))
        
        # Compare to BA with detected topic
        bin_z = (int_topic_tr == c_idx).astype(int).reshape(-1, 1)
        ba.fit(d_tr.X, bin_z, d_tr.y)
        results.append(dict(
            y_prob=ba.predict_proba(d_te.X), 
            y_pred=ba.predict(d_te.X), 
            y_true=d_te.y, 
            model="BA_discovered", 
            c_idx=c_idx,
            c_val=c_val)
        )

        for r in results:
            r["corr_tr"] = d_tr.pearsonr[0]
            r["corr_te"] = d_te.pearsonr[0]
            r["bias_tr"] = d_tr.get_bias()
            r["bias_te"] = d_te.get_bias()

        results = pd.DataFrame(results)
        with self.output().open("w") as fd:
            fd.write(results.to_json(orient="records", lines=True))

    def output(self):
        fname = "trbias={:.3f}_tebias={:.3f}_size={}_foldidx={}_trfrac={:.3f}.jsonl".format(
            self.tr_bias, self.te_bias, self.size, self.foldidx, self.tr_frac
        )
        fpath = os.path.join(
            self.outdir, "pred_benchmark", self.topic_distrib_task, fname
        )
        return luigi.LocalTarget(fpath)
