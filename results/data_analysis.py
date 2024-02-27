import pandas as pd
import numpy as np


def create_pd_from_file(filepath):
    return pd.read_csv(filepath, sep=',', header=0)


probability_per_step = create_pd_from_file("5Epochs/model_run_parameters.txt")
x = 1
x = x + 1
