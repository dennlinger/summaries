"""
Custom project implementing an alignment strategy that overfits a hierarchical encoder/decoder architecture,
which should generate attention weights that are usable for alignments.
"""
import torch.nn as nn


class OverfitAligner(nn.Module):

    def __init__(self):
        super(OverfitAligner, self).__init__()
        pass
