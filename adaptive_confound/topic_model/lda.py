from .interface import TopicModel
from sklearn.decomposition import LatentDirichletAllocation as sklLDA


class LatentDirichletAllocation(TopicModel):
    def __init__(self, k, m, model_params=None, log_params=None):
        super().__init__(k, m, model_params, log_params)
        self.tm = sklLDA(n_components=m, **self.model_params)

    def fit(self, d, *args, **kwargs):
        self.tm.fit(d.X)
        return self

    def transform(self, d, *args, **kwargs):
        return self.tm.transform(d.X)

    def get_name(self):
        if self._name is None:
            self._name = "LDA({})".format(self.m)
        return self._name
