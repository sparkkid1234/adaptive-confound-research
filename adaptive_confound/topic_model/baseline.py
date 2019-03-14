from .interface import TopicModel
from ..utils import Dataset
from sklearn.preprocessing import OneHotEncoder


class TrueTopic(TopicModel):
    def __init__(self, k):
        super().__init__(k, 2, model_params=None, log_params=None)

    def fit(self, d, *args, **kwargs):
        return self

    def transform(self, d, *args, **kwargs):
        z = d.z
        if len(z.shape) == 1:
            z = z.reshape(-1, 1)
        # ohe_z = OneHotEncoder().fit_transform(z)
        return z  # ohe_z

    def get_name(self):
        if self._name is None:
            self._name = "TrueTopic"
        return self._name
