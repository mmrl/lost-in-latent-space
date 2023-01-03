import numpy as np
import scipy as sp
import pandas as pd
from numpy import linalg as linalg

from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestRegressor

from .testing import infer
from .munkres import Munkres
from .hinton import hinton


EPS = 1e-12


def correlation(method, x, y):
    n_lats = x.shape[1]
    n_inst, n_factors = y.shape

    dummy_vars = np.random.randn(n_inst, n_lats - n_factors)
    y = np.concatenate([y, dummy_vars], axis=1)

    # Calculate correlation -----------------------------------
    if method == 'pearson':
        corr = np.corrcoef(y, x)
    elif method == 'spearman':
        corr, pvalue = sp.stats.spearmanr(y, x)
    else:
        raise ValueError('Unrecognized method {}'.format(method))
    corr = corr[:y.shape[1], y.shape[1]:]

    return corr


def latent_variable_regression(model, x, y, model_kwargs=None):
    n_lats = x.shape[1]
    n_inst, n_factors = y.shape

    for i in range(len(y.T)):
        if len(np.unique(y.T[i])) == 1:
            y[:, i] = y[:, i].astype(dtype=np.float64) + np.random.randn(len(y))

    x = (x - x.mean(axis=0)) / (x.std(axis=0) + EPS)
    y = (y - y.mean(axis=0)) / (y.std(axis=0) + EPS)

    dummy_vars = np.random.randn(n_inst, n_lats - n_factors)
    y = np.concatenate([y, dummy_vars], axis=1)

    model = init_model(model, model_kwargs)
    R = [model.fit(x, y_k).coef_ for y_k in y.T]

    return np.stack(R)


def init_model(model, model_kwargs):
    if model_kwargs is None:
        model_kwargs = {}

    if model == 'lasso':
        kwargs = {'cv': 5, 'selection': 'random', 'alphas': [0.02]}
        kwargs.update(model_kwargs)

        model = LassoCV(**kwargs)

    elif model == 'random-forest':
        model = RandomForestRegressor(**model_kwargs)

    elif model == 'logistic':
        kwargs = {'cv': 5}
        kwargs.update(model_kwargs)

        model = LogisticRegressionCV(**kwargs)

    elif model == 'isoreg':
        model = IsotonicRegression(**model_kwargs)

    else:
        raise ValueError()

    return model


class DAF:
    def __init__(self, train_data, method='spearman', assignment='max',
                 test_data=None, ignore_cat_latents=True, method_kwargs=None):
        self.train_data = train_data
        self.test_data = test_data
        self.method = method
        self.assignment = assignment
        self.method_kwargs = method_kwargs
        self.ignore_cat_latents = ignore_cat_latents

    def compute_coefficients(self, z, genfacts):
        if self.method in ['spearman', 'pearson']:
            return correlation(self.method, z, genfacts)
        elif self.method in ['lasso', 'logistic', 'isoreg']:
            return latent_variable_regression(self.method, z, genfacts,
                                              self.method_kwargs)
        elif self.method == 'projection':
            return latent_variable_regression('linreg', genfacts, z,
                                              self.method_kwargs)
        raise ValueError('Method \'{}\' not recognized'.format(self.method))

    def assign_latents(self, z, coeff, indexes=None, has_cat=False):
        dim = z.shape[1]

        if indexes is None:
            if self.assignment == 'optimal':
                indexes = Munkres().compute(-np.absolute(coeff))
                indexes = [i[1] for i in indexes]
            elif self.assignment == 'max':
                indexes = np.absolute(coeff).argmax(axis=0)

        z_sort = np.zeros(z.shape)
        for i in range(dim):
            z_sort[:, i] = z[:, indexes[i]]

        return indexes, z_sort

    def __call__(self, model, zs=None):
        if zs is None:
            z, gt = infer(model, self.train_data)

            if self.test_data is not None:
                z_test, gt_test = infer(model, self.test_data)
            else:
                z_test, gt_test = None, None
        else:
            try:
                (z, gt), (z_test, gt_test) = zs
            except ValueError:
                z, gt = zs
                z_test, gt_test = None, None

        z, gt = z.numpy(), gt.numpy()

        coeff = self.compute_coefficients(z, gt)
        idx, z_sort = self.assign_latents(z, coeff)
        coeff = self.compute_coefficients(z_sort, gt)

        if z_test is not None:
            z_test, gt_test = z_test.numpy(), gt_test.numpy()
            _, z_test_sort = self.assign_latents(z_test, coeff, idx)

            z_sort = np.concatenate([z_sort, z_test_sort])
            gt = np.concatenate([gt, gt_test])

        return DAFResults(coeff, idx, z_sort, gt,
                          list(self.train_data.factors),
                          len(self.train_data),
                          self.method, self.assignment)


class DAFResults:
    def __init__(self, coefficients, sort_idx, z_sorted, gt_vals,
                 gt_names, n_train, prediction_method, assignment_method):
        self.coefficients = coefficients
        self.sort_idx = sort_idx
        self.z_sorted = z_sorted
        self.gt_vals = gt_vals
        self.gt_names = gt_names
        self.n_train = n_train
        self.prediction_method = prediction_method
        self.assignment_method = assignment_method

    @property
    def abs_coeff(self):
        return np.abs(self.coefficients)

    @property
    def hinton_matrix(self):
        return self.abs_coeff[:self.gt_vals.shape[1]]

    def sort_embedding(self, z):
        return z[:, self.sort_idx]

    def todf(self):
        n_latents, n_factors = self.z_sorted.shape[1], self.gt_vals.shape[1]

        gt_names = ['gt {}'.format(dim) for dim in self.gt_names]
        true_latents_df = pd.DataFrame(self.gt_vals, columns=gt_names)

        z_names = (['z {}'.format(dim) for dim in self.gt_names] +
                   ['unassigned {}'.format(i) for i in range(n_latents - n_factors)])
        z_dfs = pd.DataFrame(self.z_sorted, columns=z_names)

        all_values = pd.concat([z_dfs, true_latents_df], axis=1)

        is_training = np.arange(len(self.gt_vals)) < self.n_train
        all_values['Is Training'] = is_training

        return all_values

    def score2df(self):
        di, weights = self.coefficient_entropy()
        completness = self.completeness()
        # mcc_score = self.mean_coefficient_score()

        n_latents = self.z_sorted.shape[1]

        idx = pd.MultiIndex.from_arrays([range(n_latents)], names=['latent'])
        disent_scores = pd.DataFrame(zip(di, weights), index=idx,
                                     columns=['disentanglement', 'weights'])

        idx = pd.MultiIndex.from_arrays([self.gt_names], names=['factor'])
        completness_scores = pd.DataFrame(completness, index=idx,
                                          columns=['completness'])

        return disent_scores, completness_scores, (di * weights).sum()

    def completeness(self):
        coefficients = np.abs(self.coefficients)[:self.gt_vals.shape[1]].T

        # Normalizing factors along each latent for each generative factor
        sums_d = coefficients.sum(axis=0) + EPS

        # Probabilities and entropy
        probs = coefficients / sums_d
        log_probs = np.log(probs + EPS) / np.log(coefficients.shape[0])
        entropy = - (probs * log_probs).sum(axis=0)

        # return completness scores
        return 1 - entropy

    def coefficient_entropy(self):
        """
        Disentanglement score as in Eastwood et al., 2018.
        """
        coefficients = np.abs(self.coefficients)

        n_factors = coefficients.shape[0]

        # Normalizing factors wrt to generative factors for each latent var
        sums_k = coefficients.sum(axis=0, keepdims=True) + EPS
        weights = (sums_k / sums_k.sum()).squeeze()

        # Compute probabilities and entropy
        probs = coefficients / sums_k
        log_probs = np.log(probs + EPS) / np.log(n_factors)
        entropy = - (probs * log_probs).sum(axis=0)

        # Compute scores
        di = (1 - entropy)

        return di, weights

    def disentanglement(self):
        di, weights = self.coefficient_entropy()
        return (di * weights).sum()

    def mean_coefficient_score(self):
        n_factors = self.gt_vals.shape[1]
        corr_upper_diag = np.diag(self.abs_coeff)[:n_factors]
        return corr_upper_diag.mean()

    def plot_hinton(self, ax=None):
        gt_names = [g.replace('_', ' ') for g in self.gt_names]
        latent_ids = list(range(len(self.sort_idx)))

        return hinton(self.hinton_matrix.T,
                      factor_labels=gt_names, latent_labels=latent_ids,
                      use_default_ticks=False, fontsize=11, ax=ax)


class DCIMetrics(DAF):
    def __init__(self, train_data, test_data):
        super().__init__(train_data, 'lasso', 'optimal', test_data=test_data)


class MCCScore(DAF):
    def __init__(self, train_data, test_data):
        super().__init__(train_data, 'spearman', 'optimal', test_data=test_data)
