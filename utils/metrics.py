import numpy as np


def metric(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / np.maximum(np.abs(true), 1e-5)))
    mspe = np.mean(((pred - true) / np.maximum(np.abs(true), 1e-5)) ** 2)
    return mae, mse, rmse, mape, mspe
