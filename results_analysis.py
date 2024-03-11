import numpy as np
import pandas as pd


def create_pandas_from_results(filepath):
    pandas_table = pd.read_csv(filepath, sep=",")
    return pandas_table

def evaluate_5epochs_2batches_results():
    model_parameters = create_pandas_from_results("results/5Epochs_2Batches/model_run_parameters.txt")
    student_generated = create_pandas_from_results("results/5Epochs_2Batches/student_model_generated.txt")
    x = 1
    print(x)


if __name__ == "__main__":
    evaluate_5epochs_2batches_results()
