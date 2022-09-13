"""
Compute all baselines (lead-3, lead-k, Lexrank-ST) on Klexikon.
Importantly, Klexikon already comes pre-split into sentences.
"""


from datasets import load_dataset

from summaries.baselines import lead_3, lead_k, lexrank_st

if __name__ == '__main__':
    data = load_dataset("dennlinger/klexikon")


