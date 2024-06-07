import numpy as np
import matplotlib.pyplot as plt

from plotting import *


if __name__ == '__main__':
    train_data = np.loadtxt('../Project/trainData.txt', delimiter=",").T

    train_data, label = train_data[:train_data.shape[0]-1, :], train_data[train_data.shape[0]-1, :]

    project_plot_scatter(train_data, label)

    project_plot_hist(train_data, label)

