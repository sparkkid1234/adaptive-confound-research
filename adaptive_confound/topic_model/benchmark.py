from ..utils import Dataset
from sklearn.linear_model import LogisticRegression
from ..control import BackdoorAdjustment
import pandas as pd
import numpy as np


class TopicModelBenchmark:
    def __init__(self, model_list, ba_kwargs=None, lr_kwargs=None):
        """
        Args:
            model_list (list[TopicModel]): list of initialized models
        """
        self.model_list = model_list
        if ba_kwargs is None:
            ba_kwargs = {}
        if lr_kwargs is None:
            lr_kwargs = {}
        self.ba = BackdoorAdjustment(**ba_kwargs)
        self.lr = LogisticRegression(**lr_kwargs)

    def fit_topics(self, d, list_kwargs):
        for tm, kwargs in zip(self.model_list, list_kwargs):
            # Fit the topic model
            tm.fit(d, **kwargs)
        return self

    def transform_topics(self, d):
        r = []
        for tm in self.model_list:
            r.append(tm.transform(d))
        return r

    def run(self, d_train, d_test, return_df=False):
        """
        Args:
            d (Dataset)
        """
        results = []

        # Loop over topic models
        for tm in self.model_list:
            # Retrieve the most significant topic for each instance
            z_train = tm.transform(d_train)
            m = np.shape(z_train)[1]

            if m == 1:
                int_z_train = np.array(z_train).astype(int)
            else:
                int_z_train = np.argmax(z_train, axis=1)

            # Control for each topic individually (one-vs-all scheme)
            # using backdoor adjustment and evaluate the performance
            for z_idx, zi in enumerate(range(m)):
                if m == 1:
                    bin_z_train = int_z_train
                else:
                    bin_z_train = (int_z_train == zi).astype(int).reshape(-1, 1)
                self.ba.fit(d_train.X, bin_z_train, d_train.y)
                y_pred = self.ba.predict(d_test.X)
                result_entry = {
                    "y_pred": y_pred,
                    "y_true": d_test.y,
                    "topic_model": tm.get_name(),
                    "classifier": "BackdoorAdjustment",
                    "z_i": z_idx,
                }
                results.append(result_entry)

        # Use logistic regression as a baseline
        self.lr.fit(d_train.X, d_train.y)
        y_pred = self.lr.predict(d_test.X)
        result_entry = {
            "y_pred": y_pred,
            "y_true": d_test.y,
            "topic_model": None,
            "classifier": "LogisticRegression",
            "z_i": 0,
        }
        results.append(result_entry)

        # return the benchmark results as a pandas DataFrame
        if return_df:
            results = pd.DataFrame(results)
        return results
