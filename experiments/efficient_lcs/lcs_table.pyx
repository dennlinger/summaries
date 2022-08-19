"""
Cython extension for lcs table creation with static data types.
"""

import numpy as np

def _cython_lcs_table(ref, can):
    """Create 2-d LCS score table."""
    rows = len(ref)
    cols = len(can)
    lcs_table = np.zeros([rows + 1, cols + 1], dtype=np.int)
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if ref[i - 1] == can[j - 1]:
                lcs_table[i, j] = lcs_table[i - 1, j - 1] + 1
            else:
                lcs_table[i, j] = max(lcs_table[i - 1, j], lcs_table[i, j - 1])
    return lcs_table