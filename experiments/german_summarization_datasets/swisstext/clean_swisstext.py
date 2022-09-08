"""
Script to work with the summarization data for the Swisstext summarization challenge.
Notably, this includes only a training and test set, but no validation set.
Also, since the copyright of this dataset is unclear, there exists no processed version on Huggingface.
"""
import os
import csv

import pandas as pd


if __name__ == '__main__':

    base_path = "/home/dennis/swisstext/"

    train_set = pd.read_csv(os.path.join(base_path, "data_train.csv"), delimiter=",").to_dict("records")
