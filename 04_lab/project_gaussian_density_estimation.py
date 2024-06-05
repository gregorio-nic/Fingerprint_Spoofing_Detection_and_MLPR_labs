import numpy as np
import matplotlib.pyplot as plt
import sys

from gaussian_density_estimation import *

def plot_heatmap(data, C_ML, color="Greys"):

    corr_matrix = np.abs(np.corrcoef(data))

    print(corr_matrix-C_ML.max())


    plt.figure()
    plt.imshow(corr_matrix, cmap=color, interpolation='nearest')
    plt.xticks(np.arange(corr_matrix.shape[1]))
    plt.yticks(np.arange(corr_matrix.shape[1]))
    plt.colorbar()  # Aggiunge una barra dei colori per la scala
    plt.show(block=False)

    plt.close()


if __name__ == '__main__':
    #np.set_printoptions(threshold=sys.maxsize)

    train_data = np.loadtxt('../Project/trainData.txt', delimiter=',').T

    train_data, label = train_data[:train_data.shape[0]-1, :], train_data[train_data.shape[0]-1, :]
    #print('train_data: ', train_data, train_data.shape)
    #print('label: ', label, label.shape)
    #print(train_data[:, 0])

    m_ML = vcol(np.mean(train_data, axis=1))
    C_ML = 1/train_data.shape[1] * (train_data-m_ML) @ (train_data-m_ML).T

    plot_heatmap(train_data, C_ML, color='Greys')

    train_data_0 = train_data[:, label == 0]
    m_ML_0 = vcol(np.mean(train_data_0, axis=1))
    C_ML_0 = 1 / train_data_0.shape[1] * (train_data_0 - m_ML_0) @ (train_data_0 - m_ML_0).T

    train_data_1 = train_data[:, label == 1]
    m_ML_1 = vcol(np.mean(train_data_1, axis=1))
    C_ML_1 = 1 / train_data_1.shape[1] * (train_data_1 - m_ML_1) @ (train_data_1 - m_ML_1).T

    #print('0: ', train_data_0, train_data_0.shape)
    #print('1: ', train_data_1, train_data_1.shape)

    print(train_data_0.shape[1] - train_data_1.shape[1])

    plot_heatmap(train_data_0, C_ML, color='Blues')
    plot_heatmap(train_data_1, C_ML, color='Reds')


    # plot gaussian density estimation whole dataset for each feature
    for f in range(train_data.shape[0]):
        plt.figure()
        train_plot = np.linspace(train_data[f,:].min(axis=0), train_data[f,:].max(axis=0), 1000)
        print(np.exp(logpdf_GAU_ND(vrow(train_plot), m_ML[f, :], vcol(C_ML[f, f]))))
        plt.plot(train_plot.ravel(), np.exp(logpdf_GAU_ND(vrow(train_plot), m_ML[f,:], vcol(C_ML[f, f]))), color='red')
        plt.hist(train_data[f, :].ravel(), bins=50, density=True)
        plt.title(f'Whole dataset gaussian density estimation - Feature {f}')
        plt.show()

    # plot gaussian density estimation label 0 for each feature
    for f in range(train_data_0.shape[0]):
        plt.figure()
        train_plot = np.linspace(train_data_0[f, :].min(axis=0), train_data_0[f, :].max(axis=0), 1000)
        print(np.exp(logpdf_GAU_ND(vrow(train_plot), m_ML_0[f, :], vcol(C_ML_0[f, f]))))
        plt.plot(train_plot.ravel(), np.exp(logpdf_GAU_ND(vrow(train_plot), m_ML_0[f, :], vcol(C_ML_0[f, f]))),
                 color='red')
        plt.hist(train_data_0[f, :].ravel(), bins=50, density=True)
        plt.title(f'Label 0 gaussian density estimation - Feature {f}')
        plt.show()

    # plot gaussian density estimation label 1 for each feature
    for f in range(train_data_1.shape[0]):
        plt.figure()
        train_plot = np.linspace(train_data_1[f, :].min(axis=0), train_data_1[f, :].max(axis=0), 1000)
        print(np.exp(logpdf_GAU_ND(vrow(train_plot), m_ML_1[f, :], vcol(C_ML_1[f, f]))))
        plt.plot(train_plot.ravel(), np.exp(logpdf_GAU_ND(vrow(train_plot), m_ML_1[f, :], vcol(C_ML_1[f, f]))),
                 color='red')
        plt.hist(train_data_1[f, :].ravel(), bins=50, density=True)
        plt.title(f'Label 1 gaussian density estimation - Feature {f}')
        plt.show()
