import luigi
import os.path
import numpy as np
import pickle
from ..utils import *


class DatasetPairs(luigi.Task):
    """
    Outputs a list of training/testing dataset pairs.
    """

    d = luigi.Parameter()
    tr_bias = luigi.FloatParameter()
    te_bias = luigi.FloatParameter()
    foldidx = luigi.IntParameter()
    outdir = luigi.Parameter()
    tr_frac = luigi.FloatParameter(default=.5)
    random_seed = luigi.IntParameter(default=1234)
    size = luigi.IntParameter(default=1000)

    def requires(self):
        pass

    def run(self):
        # reads original dataset
        data = read_pickle(self.d)

        # creates train and test unbiased datasets
        seed = self.random_seed + self.foldidx
        np.random.seed(seed)

        ogsize = data.X.shape[0]
        data_range = list(range(ogsize))
        np.random.shuffle(data_range)
        spliti = int(ogsize * self.tr_frac)
        tr_split = data_range[:spliti]
        te_split = data_range[spliti:]

        tr_data = Dataset(X=data.X[tr_split], y=data.y[tr_split], z=data.z[tr_split])
        te_data = Dataset(X=data.X[te_split], y=data.y[te_split], z=data.z[te_split])

        # creates biased datasets
        biased_pair = [
            tr_data.make_confounding_dataset(self.tr_bias, self.size),
            te_data.make_confounding_dataset(self.te_bias, self.size),
        ]
        for d in biased_pair:
            d.parent = None

        # writes out datasets
        for d, outpath in zip(biased_pair, self.output()):
            with outpath.open("w") as fd:
                pickle.dump(d, fd)

    def output(self):
        fname = "trbias={:.3f}_tebias={:.3f}_size={}_foldidx={}_trfrac={:.3f}".format(
            self.tr_bias, self.te_bias, self.size, self.foldidx, self.tr_frac
        )
        fpath = os.path.join(self.outdir, "datapairs", fname)
        fpaths = [_.format(fpath) for _ in ["{}_train.pkl", "{}_test.pkl"]]

        return [luigi.LocalTarget(_, format=luigi.format.Nop) for _ in fpaths]
