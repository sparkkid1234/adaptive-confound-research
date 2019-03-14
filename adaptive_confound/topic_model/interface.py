from datetime import datetime
from abc import ABC, abstractmethod
import os.path


class TopicModel(ABC):
    """
    Interface used to implement a topic model strategy.
    """

    def __init__(self, k, m, model_params=None, log_params=None):
        """
        Initialize the topic model.

        Args:
            k (int): The number of features expected by the model.
            m (int): The number of topics to generate.
            model_params (dict, optional): Dictionary used to pass configuration parameters to the model such as the number of units per hidden layer in a deep neural model or the perplexity value in LDA.
            log_params (dict, optional): Optional dictionary to store the logging information to use with TensorBoard.
        """
        self.k = k
        self.m = m
        self.model_params = model_params if model_params is not None else {}
        if log_params is None:
            log_params = {"write_graph": True}
        elif "write_graph" not in log_params.keys():
            log_params["write_graph"] = True
        self.log_params = log_params
        self._name = None

        # Create path to log the training info for this model
        self._create_log_path()

    def _create_log_path(self):
        """
        Creates the path to store the logs given a model name and add it to the logging information.
        
        Args:
            name (str): The name of the topic model technique.
            
        Returns:
            log_dir (str): The path to the logging file for this model.
        """
        name = self.get_name()
        log_dir = self.log_params.get("log_dir", "./")
        dt = datetime.now().strftime("%x_%X")

        log_dir = os.path.join(log_dir, "{}@{}".format(name, dt))
        self.log_params["log_dir"] = log_dir
        return log_dir

    @abstractmethod
    def fit(self, d, *args, **kwargs):
        """
        Fits the implemented topic model technique to the data.
        
        Args:
            d (Dataset): An instance of the Dataset class.
            m (int): The number of topics to use.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            tm (TopicModel): The current object after the model has been fit.
        """
        pass

    @abstractmethod
    def transform(self, d, *args, **kwargs):
        """
        Given a dataset containing `n` documents, this method returns
        the distribution of `m` topics for each document.
        
        Args:
            d (Dataset): An instance of the Dataset class.
            
        Returns:
            td (numpy.ndarray): The distribution of topics for each documents as a Numpy array of shape (n, m).
        """
        pass

    @abstractmethod
    def get_name(self):
        """
        Returns the name of the model.
        """
        pass
