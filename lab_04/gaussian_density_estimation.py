import numpy as np
import matplotlib.pyplot as plt

'''turns vector into row-matrix'''


def vrow(vector):
    return vector.reshape(1, vector.size)


'''turns vector into col-matrix'''


def vcol(vector):
    return vector.reshape(vector.size, 1)


def loglikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum(0)


'''log density for a data matrix X[M, N]'''


def logpdf_GAU_ND(X, mu, C):
    P = np.linalg.inv(C)
    return -0.5 * X.shape[0] * np.log(np.pi * 2) - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * (
            (X - mu) * (P @ (X - mu))).sum(0)


'''log density for single sample x=np.array([M, 1])'''


def logpdf_GAU_ND_1_sample(x, mu, C):
    M = mu.shape[0]
    return - (M / 2) * np.log(2 * np.pi) - (1 / 2) * np.linalg.slogdet(C)[1] - (1 / 2) * (x - mu).T @ np.linalg.inv(
        C) @ (x - mu)


if __name__ == '__main__':
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1, 1)) * 1.0
    C = np.ones((1, 1)) * 2.0

    # plt.figure()
    # plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    # plt.show()

    # Checking 1D solution
    pdfSol = np.load('Solution/llGAU.npy')
    pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)

    print(np.abs(pdfSol - pdfGau).max())

    # Checking multi-dim solution
    XND = np.load('Solution/XND.npy')
    muND = np.load('Solution/muND.npy')
    CND = np.load('Solution/CND.npy')

    pdfSol = np.load('Solution/llND.npy')
    pdfGau = logpdf_GAU_ND(XND, muND, CND)
    print(np.abs(pdfSol - pdfGau).max())

    # Computing multi-dim maximum likelihood
    m_ML = vcol(np.mean(XND, axis=1))
    C_ML = 1 / XND.shape[1] * (XND - m_ML) @ (XND - m_ML).T

    ll = loglikelihood(XND, m_ML, C_ML)
    print(ll)

    # Computing single-dim maximum likelihood
    X1D = np.load("Solution/X1D.npy")
    m_ML_1D = np.mean(X1D, axis=1)
    C_ML_1D = 1 / X1D.shape[1] * (X1D - m_ML_1D) @ (X1D - m_ML_1D).T

    print(m_ML_1D, C_ML_1D)

    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot_1D = np.linspace(-8, 12, 1000)
    plt.plot(XPlot_1D.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot_1D), m_ML_1D, C_ML_1D)))
    plt.show()

    ll_1D = loglikelihood(X1D, m_ML_1D, C_ML_1D)
    print(ll_1D)


    print(m_ML_1D.shape, C_ML_1D.shape)