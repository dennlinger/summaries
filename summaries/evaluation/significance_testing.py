"""
A "collection" of tests for significance testing
"""
from typing import List, Callable

import numpy as np


def paired_bootstrap_test(gold_labels: List,
                          system_a: List,
                          system_b: List,
                          scoring_function: Callable,
                          n_resamples: int = 10_000,
                          seed: int = 256) -> float:
    """
    Method to compute a paired bootstrap resampling significance test.
    It will be tested whether system A is significantly better than system B and return the p-value.
    Re-samples a temporary test set of the same size as the original, but with replacements.
    If the score of system A is still better than that of system B, we count it as a "success" and repeat n times.
    This implementation largely follows Philipp Koehn's 2004 work
    "Statistical Significance Tests for Machine Translation Evaluation", see
    https://aclanthology.org/W04-3250/

    :param gold_labels: List of ground truth labels/values for a test set.
    :param system_a: List of predictions of system A on the test set. Assumed to be the "better" system."
    :param system_b: List of predictions of system B on the test set. Assumed to be the "baseline" method.
    :param scoring_function: An arbitrary evaluation function which takes in two lists (system, gold) and produces
        an evaluation score (singular float).
    :param n_resamples: Number of times to re-sample the test set.
    :param seed: Random seed to ensure reproducible results.
    :return: p-value of the statistical significance that A is better than B.
    """

    if len(gold_labels) != len(system_a) != len(system_b):
        raise ValueError("Ensure that gold and system outputs have the same lengths!")

    number_of_times_a_better_b = 0
    equal_scores = 0

    rng = np.random.default_rng(seed=seed)

    # Cast to np.array for easier index access
    gold_labels = np.array(gold_labels)
    system_a = np.array(system_a)
    system_b = np.array(system_b)

    for _ in range(n_resamples):
        # Perform re-sampling with replacement of a similarly large "test set".
        indices = rng.choice(len(gold_labels), len(gold_labels), replace=True)

        curr_gold = gold_labels[indices]
        curr_a = system_a[indices]
        curr_b = system_b[indices]

        # Compute the system evaluation scores under the altered test set
        score_a = scoring_function(curr_a, curr_gold)
        score_b = scoring_function(curr_b, curr_gold)

        # TODO: Investigate whether strict improvements should be counted?
        if score_a > score_b:
            number_of_times_a_better_b += 1

        if score_a == score_b:
            equal_scores += 1

    if equal_scores > 0:
        print(f"Encountered samples in which scores were equal, which are not counted in the returned p-value.\n"
              f"If cases with equal scoring were to be considered a 'win' for system A over B, then the corrected "
              f"p-value would be {1 - ((number_of_times_a_better_b + equal_scores) / n_resamples)}.")

    p_value = 1 - (number_of_times_a_better_b / n_resamples)
    return p_value
