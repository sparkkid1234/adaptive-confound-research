import numpy as np
import pandas as pd
import sys
import logging
import pickle
import scipy.sparse as sp
from scipy.stats import pearsonr
from tqdm import tqdm
from types import SimpleNamespace
from scipy import sparse

class Dataset(SimpleNamespace):
    """
    Simple class to encapsulate a dataset component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # flag indicating that the dataset is biased
        self.biased = False

    def _has(self, attr):
        return attr in self.__dict__.keys()

    def _check_xyz_exist(self):
        try:
            X = self.X
            y = self.y
            z = self.z
            
            return X, y, z
        except AttributeError as e:
            logging.error(
                "Make sure the current dataset has the following attributes: X, y, and z."
            )
            raise e
    
    def _check_biased_datasets_exist(self):
        try:
            bd = self.biased_datasets
        except AttributeError as e:
            if create:
                self.create_biased_datasets(size=1000, biases=[.1, .3, .5, .7, .9])
                bd = self.biased_datasets
            else:
                logging.error(
                    "Make sure the biased_datasets attribute exists in your current dataset by running d.create_biased_datasets."
                )
                raise e
        return bd
    
    def _make_confounding_dataset(self, X, y, z, bias, size, pos_prob=None, zpos_prob=None):
        """
        Create Sample a dataset of given size where c is a confounder for y with strength=bias. We take care not to introduce selection bias (that is, p(z=1) is representative of training data). This assumes that #[z=1] < #[y=1].

        Args:
            X: data matrix
            y: class labels (0, 1)
            z: confounding labels (0,1)
            bias: amount of bias (0-1)
            size: number of samples
            pos_prob: proportion of instances where y=1. If None, this probability is computed from the `y` parameter.

        Returns:
            d (Dataset): A biased subsample of the current dataset packed in a Dataset instance.
        """
        if pos_prob is None:
            pos_prob = 1. * sum(y) / len(y)

        both_pos = [i for i in range(len(y)) if y[i] == 1 and z[i] == 1]
        both_neg = [i for i in range(len(y)) if y[i] == 0 and z[i] == 0]
        ypos_zneg = [i for i in range(len(y)) if y[i] == 1 and z[i] == 0]
        yneg_zpos = [i for i in range(len(y)) if y[i] == 0 and z[i] == 1]

        for x in [both_pos, both_neg, yneg_zpos, ypos_zneg]:
            np.random.shuffle(x)

        # if bias=.9, then 90% of instances where c=1 will also have y=1
        # similarly, 10% of instances where c=1 will have y=0
        zprob = zpos_prob if zpos_prob is not None else 1. * sum(z) / len(z)
        n_zpos = int(zprob * size)
        n_zneg = size - n_zpos
        n_ypos = int(pos_prob * size)
        n_yneg = size - n_ypos

        n_11 = int(bias * n_zpos)
        n_01 = int((1 - bias) * n_zpos)
        n_10 = n_ypos - n_11
        n_00 = n_yneg - n_01
        if n_10 < 0 or n_00 < 0:
            logging.warning(
                "Bias argument is too large for this dataset so the size of the resulting dataset will be larger than wanted and both p(y) and p(z) will not be the same compared to your original dataset. Possible solutions are to reduce the amount of bias to inject or make sure that your original dataset is balanced on y and z (i.e. p(y) = p(z) = 0.5)."
            )

        a_11 = both_pos[:n_11]
        a_00 = both_neg[:n_00]
        a_10 = ypos_zneg[:n_10]
        a_01 = yneg_zpos[:n_01]

        r = np.array(a_11 + a_00 + a_10 + a_01)
        np.random.shuffle(r)
        
        # Create biased dataset object
        d = Dataset(X=X[r], y=y[r].flatten(), z=z[r].flatten(), indices=list(r))
        d.biased = True
        d.pearsonr = pearsonr(d.y, d.z)

        return d

    def make_zbiased_dataset(self, zprob, size, pos_prob=None):
        X, y, z = self._check_xyz_exist()
        
        d = self._make_confounding_dataset(X, y, z, self.get_bias(), size, pos_prob, zprob)
        d.parent = self
        if self._has("features"):
            d.features = self.features
            
        return d
    
    def make_confounding_dataset(
        self, bias, size, pos_prob=None
    ):
        """
        Create Sample a dataset of given size where c is a confounder for y with strength=bias. We take care not to introduce selection bias (that is, p(z=1) is representative of training data). This assumes that #[z=1] < #[y=1].

        Args:
            bias: amount of bias (0-1)
            size: number of samples
            pos_prob: proportion of instances where y=1. If None, this probability is computed from the `y` parameter.

        Returns:
            d (Dataset): A biased subsample of the current dataset packed in a Dataset instance.
            
        Raises:
            AttributeError when the dataset has not been correctly initialized with one of X, y, and z.
        """
        X, y, z = self._check_xyz_exist()
        
        d = self._make_confounding_dataset(X, y, z, bias, size, pos_prob)
        d.parent = self
        if self._has("features"):
            d.features = self.features
        return d

    def create_biased_datasets(self, size, biases, tr_frac=.5, k=1, rand_seed=1234):
        """
        Creates multiple biased training and testing datasets.
        First, the original data is split into training and testing based on the value of `tr_frac`.
        Then, for the two splits, `k` biased datasets are created for each value in `biases`.

        Args:
            size (int): The number of instances in each generated dataset.
            biases (list[float]): The multiple bias values used to generate the datasets (0-1).
            tr_frac (float, optional): The fraction of the original dataset used for training (default: 0.5).
            k (int, optional): The number of datasets generated for every possible bias (default: 1).
            rand_seed (int, optional): The random seed to feed numpy in order to create reproducible experiments (default: 1234).

        Returns:
            df (pandas.DataFrame): A pandas DataFrame containing the info on the generated datasets.
        """
        X, y, z = self._check_xyz_exist()
        
        np.random.seed(rand_seed)

        ogsize = X.shape[0]
        data_range = list(range(ogsize))
        np.random.shuffle(data_range)
        spliti = int(ogsize * tr_frac)
        tr_split = data_range[:spliti]
        te_split = data_range[spliti:]

        self.biased_datasets = []

        for subidx, kind in zip([tr_split, te_split], ["train", "test"]):
            for b in tqdm(
                biases, desc="Generate biased {} datasets".format(kind), leave=False
            ):
                for ik in range(k):
                    d = self._make_confounding_dataset(
                        X[subidx], y[subidx], z[subidx], b, size
                    )
                    d.kind = kind
                    d.parent = self
                    if self._has("features"):
                        d.features = self.features
                    self.biased_datasets.append(d)

    def _make_balanced_dataset(self, X, y, z):
        idx = [
            [i for i in range(len(y)) if y[i] == 1 and z[i] == 1],
            [i for i in range(len(y)) if y[i] == 0 and z[i] == 0],
            [i for i in range(len(y)) if y[i] == 1 and z[i] == 0],
            [i for i in range(len(y)) if y[i] == 0 and z[i] == 1],
        ]

        # Subsample to make p(y) and p(z) balanced
        minsize = min([len(x) for x in idx])
        for x in idx:
            np.random.shuffle(x)
        bidx = sum([x[:minsize] for x in idx], [])
        np.random.shuffle(bidx)

        # Create balanced dataset
        d = Dataset(X=X[bidx], y=y[bidx], z=z[bidx], indices=bidx)

        return d

    def make_balanced_dataset(self):
        d = self._make_balanced_dataset(self.X, self.y, self.z)
        d.parent = self
        if self._has("features"):
            d.features = self.features
        return d

    def get_py(self):
        if not self._has("_py"):
            self._py = np.sum(self.y) / len(self.y)
        return self._py

    def get_pz(self):
        if not self._has("_pz"):
            self._pz = np.sum(self.z) / len(self.z)
        return self._pz

    def get_bias(self):
        if not self._has("_bias"):
            n_11 = len([i for i in range(len(self.y)) if self.y[i] == 1 and self.z[i] == 1])
            n_01 = len([i for i in range(len(self.y)) if self.y[i] == 0 and self.z[i] == 1])
            self._bias = n_11 / (n_11 + n_01)
        return self._bias

    def get_size(self):
        if not self._has("_size"):
            self._size = self.X.shape[0]
        return self._size
    
    def get_biased_datasets(self, kind, create=False):
        """
        Args:
            kind (str): String indicating if we want to return the train or the test datasets.
            create (bool, optional): When set to True, the biased dataset will be created if they do not already exist with the following default arguments: size=1000 and biases=[.1,.3,.5,.7,.9]. Default to False.
        
        Raises:
            AttributeError
            ValueError
        """
        bd = self._check_biased_datasets_exist()
        if kind not in ["train", "test"]:
            raise ValueError("The variable kind must be one of ['train', 'test'].")
        return [d for d in bd if d.kind == kind]

    def iterate_biased_datasets(self, kind, create=False):
        """
        Args:
            kind (str): String indicating if we want to iterate over the train or the test datasets.
            create (bool, optional): When set to True, the biased dataset will be created if they do not already exist with the following default arguments: size=1000 and biases=[.1,.3,.5,.7,.9]. Default to False.
        
        Raises:
            AttributeError
            ValueError
        """
        bd = self._check_biased_datasets_exist()
        if kind not in ["train", "test"]:
            raise ValueError("The variable kind must be one of ['train', 'test'].")
        for d in bd:
            if d.kind == kind:
                yield d

    def __repr__(self):
        repr = "< Dataset: size={}, p(y)={:.2f}, p(z)={:.2f}, bias={:.2f}, parent={} >"
        parent = None
        if self._has("parent"):
            parent = self.parent
        return repr.format(
            self.get_size(), self.get_py(), self.get_pz(), self.get_bias(), parent
        )

    def to_pickle(self, path, with_parent=True):
        """
        Writes the dataset to file using python pickling.
        
        Args:
            path (str): Path to the pickle file to use to save the dataset
        """
        dumped = None
        if not with_parent:
            tmp_parent = self.parent
            self.parent = None
        
        with open(path, "wb") as fd:
            dumped = pickle.dump(self, fd)

        if not with_parent:
            self.parent = tmp_parent
        
        return dumped


def read_pickle(path):
    """
    Reads a file and unpickles it.
    """
    with open(path, "rb") as fd:
        return pickle.load(fd)


def concat(dataset_list):
    d = Dataset()
    if np.any([sp.issparse(_.X) for _ in dataset_list]):
        d.X = sp.vstack([_.X for _ in dataset_list])
    else:
        d.X = np.vstack([_.X for _ in dataset_list])
    d.y = np.hstack([_.y for _ in dataset_list]).flatten()
    d.z = np.hstack([_.z for _ in dataset_list]).flatten()
    
    has_features = [d._has("features") for d in dataset_list]
    if np.any(has_features):
        fts_idx = np.where(has_features)[0][0]
        d.features = dataset_list[fts_idx].features
    d.parent = dataset_list
    return d
