"""
Tests for testing functions
"""
import unittest

import numpy as np
from scipy.stats import pearsonr

from summaries.evaluation import paired_bootstrap_test, permutation_test

def pearson(x, y):
    """
    Wrapper function around pearson correlation that only returns the correlation score.
    """
    return pearsonr(x, y)[0]


class TestEvaluation(unittest.TestCase):

    def test_paired_bootstrap_test(self):
        rng = np.random.default_rng(seed=54)

        x = rng.integers(0, 5, size=64)
        y = rng.integers(0, 5, size=64)
        gold = rng.integers(0, 6, size=64)

        paired_bootstrap_test(gold, x, y, pearson, n_resamples=100)

    def test_permutation_test(self):

        rng = np.random.default_rng(seed=54)

        x = rng.integers(0, 5, size=64)
        y = rng.integers(0, 5, size=64)
        gold = rng.integers(0, 6, size=64)

        permutation_test(gold, x, y, pearson, n_resamples=100)
