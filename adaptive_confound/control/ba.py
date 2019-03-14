from sklearn.linear_model import LogisticRegression
from scipy import sparse
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from collections import Counter, defaultdict
import numpy as np
import itertools as it


class BackdoorAdjustment:
    """
    Implements a back-door adjustment classifier. This classifier aims to adjust
    for confounding bias. It assumes that the confounding variables are known at
    training time but not at testing time. It uses Pearl's back-door adjustment
    to adjust for these variables.
    """

    def __init__(self, transform=lambda _: _, pz_method="independency"):
        """
        Creates a BackdoorAdjustment classifier.

        Args:
            
            transform (function, optional): Function applied to the features
            matrix that encodes the confounding variable. Defaults to the
            identity function.

            pz_method (str, optional): String in ['independency', 'pairwise',
            'prior']. If set to 'independency', we compute p(z) as the product
            of p(z_0)*p(z_1)*...*p(z_n) therefore assuming that the components
            of z are independent. If set to 'pairwise', we build a pairwise
            Markov network z_0 -- z_1 -- ... -- z_n-1 -- z_n -- z_0. Each pair
            of variables is associated with a factor and the joint probability
            is the normalized factor product. Finally, if this argument is set
            to 'prior' then we directly compute the prior joint distribution
            p(z_0, z_1, ..., z_n) from the data: this requires that 2^n training
            instances when the z_i's are binary.

        """
        self.clf = LogisticRegression(class_weight="balanced")
        self.transform = transform
        self.pz_method = pz_method

        self.pz_func_dict = {
            "independency": self.__pz_independency,
            "pairwise": self.__pz_pairwise,
            "prior": self.__pz_prior,
        }

    def __pz_independency(self, z_features):
        """
        This method of computing p(z) assumes that all z_i are independent,
        therefore $p(z) = \prod_{i=0}^{k}{p(z_i)}$.
        """

        sum_z = z_features.sum(axis=0)
        sum_z = np.ravel(sum_z)

        probas_z = []
        index_z = []

        for ci in range(self.dim_z):
            b, e = self.ohe.feature_indices_[[ci, ci + 1]]
            idxz_i = [str(_) for _ in range(e - b)]

            sz_i = sum_z[b:e]
            sz_i += 1  # smoothing
            pz_i = sz_i / np.sum(sz_i)
            index_z.append(idxz_i)
            probas_z.append(pz_i)

        enc_idx = self.ohe.transform(list(it.product(*index_z))).toarray()
        idx_iter = ["".join(str(int(_)) for _ in r) for r in enc_idx]
        # idx_iter  = ("".join(_) for _ in it.product(*index_z))
        prob_iter = (np.product(_) for _ in it.product(*probas_z))

        z_proba = defaultdict(float)
        z_proba.update({k: v for k, v in zip(idx_iter, prob_iter)})
        return z_proba

    def __pz_pairwise(self, z_features):
        """
        This method creates the pairwise graph of the confounding variables and
        compute p(z) as the product $\frac{1}{K}\prod_{i \neq j}{\phi_{i,j}}$
        where $\phi_{i,j}$ is the factor for $z_i$ and $z_j$.
        """
        factors = defaultdict(int)
        factors_idx = list(it.combinations(range(self.dim_z), 2))
        dim_zi = [0] * self.dim_z

        for ci, cj in factors_idx:
            # retrieve indices for columns encoding confounders i and j
            bi, ei = self.ohe.feature_indices_[[ci, ci + 1]]
            bj, ej = self.ohe.feature_indices_[[cj, cj + 1]]
            dim_zi[ci] = ei - bi
            dim_zi[cj] = ej - bj

            # retrieve confounders i and j, and concatenate them
            z_i = z_features[:, bi:ei]
            z_j = z_features[:, bj:ej]
            z_ij = sparse.hstack([z_i, z_j])

            # compute factor for confounders i and j
            z_ij_str = z_ij.astype(int).toarray().astype(str)
            factors[ci, cj] = Counter("".join(r) for r in z_ij_str)

        # print(factors)

        z_possible_values = (
            self.ohe.transform(list(it.product(*[range(_) for _ in dim_zi])))
            .toarray()
            .astype(int)
        )

        z_proba = defaultdict(float)
        z_proba_keys = []
        z_proba_values = []

        for row in z_possible_values:
            row_factors = []
            row_str = "".join(str(_) for _ in row)

            for ci, cj in factors_idx:
                bi, ei = self.ohe.feature_indices_[[ci, ci + 1]]
                bj, ej = self.ohe.feature_indices_[[cj, cj + 1]]

                subrow_str = row_str[bi:ei] + row_str[bj:ej]
                f = factors[(ci, cj)][subrow_str] + 1.  # factor + smoothing

                row_factors.append(f)

            z_proba_keys.append(row_str)
            z_proba_values.append(np.product(row_factors))
            if np.product(row_factors) < 0:
                raise OverflowError(
                    "Too many confounding variables or factors too large to be represented."
                )

        z_proba_sum = np.sum(z_proba_values)
        z_proba.update(
            {k: v / z_proba_sum for k, v in zip(z_proba_keys, z_proba_values)}
        )

        return z_proba

    def __pz_prior(self, z_features):
        """
        This method computes $p(z) = p(z_0, z_1, ..., z_k)$ directly from the
        training data using MLE. This requires a large number of training
        instances.
        """
        z_count = Counter(
            "".join(str(int(_)) for _ in z_features.getrow(i).toarray().ravel())
            for i in range(self.n)
        )

        z_proba = defaultdict(float)
        z_proba.update({k: v / self.n for k, v in z_count.items()})
        return z_proba

    def predict_proba(self, X):
        """
        Predicts the probability of each class for all the instances.

        Args:
            X (matrix like object): features matrix of the instances to predict.

        Returns:
            numpy.array: array of n rows and m columns.

        """
        # p(y|do(x)) = sum_z{p(y|x,z)*p(z)}
        len_x = X.shape[0]

        # compute exp(- theta_t . x_t)
        theta_t = self.clf.coef_[0][: self.dim_x]
        exp_t = np.exp(-(self.clf.intercept_ + X.dot(theta_t))).reshape(-1, 1)

        # compute p(y|x,z)
        p_ygxz = 1 / (1 + exp_t.dot(self.exp_c))

        # compute p(y|do(x)) = sum_z{p(y|x,z) * p(z)}
        p_ygdox = p_ygxz.dot(self.pz_te)

        # transform to expected format
        p_ygdox = np.column_stack([1 - p_ygdox, p_ygdox])

        return p_ygdox

    def predict(self, X):
        """
        Computes the most likely class for each instance.

        Args:
            X (matrix like object): features matrix of the instances to predict.

        Returns:
            numpy array of dimensions (n,).

        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def fit(self, X, Z, y, sample_weight=None):
        """
        Trains a classifier given a features matrix X, a list of labels y, and a confounders matrix z.

        Args:
            X (matrix like object): features matrix of dimensions (n,m) for n
            instances and m features.
            Z (matrix like object): confounders matrix of dimensions (n,p) for
            n instances and p confounders.
            y (list): annotation of each instance in X.

        Returns:
            LogisticRegression classifier fit on the adjusted data.
        """
        len_x, dim_x = np.shape(X)
        len_y = len(y)
        len_z, dim_z = np.shape(Z)

        if len_x != len_y:
            raise ValueError(
                "Vector y must be a of dimension (n,1) where n is the number of training instances. Reshape it using np.reshape(y, (-1,1))."
            )
        if len_x != len_z:
            raise ValueError(
                "X and Z must have the same number of rows. If Z is a list, reshape it using np.reshape(Z, (-1,1))."
            )

        self.n = len_x
        self.dim_x = dim_x
        self.dim_y = np.unique(y).shape[0]
        self.dim_z = dim_z

        # Encode Z with OneHotEncoder and apply the features transformation
        self.ohe = OneHotEncoder()
        z_features = self.transform(self.ohe.fit_transform(Z))

        # compute p(z)
        pz_func = self.pz_func_dict[self.pz_method]
        self.z_proba = pz_func(z_features)

        # append the confounder column and fit on the data
        Xz = sparse.hstack((X, z_features))

        # easy access to classifier coefficients
        self.clf = self.clf.fit(Xz, np.ravel(y), sample_weight=sample_weight)
        self.coef_ = np.array([self.clf.coef_[0][: -z_features.shape[1]]])

        # create probability of z for prediction time
        self.z_te = np.array([[int(_) for _ in l] for l in self.z_proba.keys()])
        self.pz_te = np.array(list(self.z_proba.values()))

        # compute exp(- theta_c . x_c) that will be used at prediction time
        theta_c = self.clf.coef_[0][-self.z_te.shape[1] :]
        x_c = self.z_te
        self.exp_c = np.exp(-theta_c.dot(x_c.T)).reshape(1, -1)

        return self
