"""
common contains the common functionailities that
are useful during the training phase.
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def train_arima_model(train_df: pd.DataFrame, p: float, d: float, q: float) -> ARIMA:
    """
    train an ARIMA model using the given order pramaters
    and the train dataset.
    """
    order = (p, d, q)
    model = ARIMA(train_df["price"], order=order)
    return model.fit()


def split_train_test(
    df: pd.DataFrame, percentage: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    simply split the given dataset into the train and test.
    """
    train_size = int(len(df) * percentage)
    train, test = df[:train_size], df[train_size:]
    # these asserts are here to be compatible with type checking
    # system
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    return train, test
