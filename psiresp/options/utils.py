import pandas as pd

def read_csv(path):
    return pd.read_csv(path, index_col=0, header=0)


def load_text(file):
    with open(file, "r") as f:
        return f.read()
        