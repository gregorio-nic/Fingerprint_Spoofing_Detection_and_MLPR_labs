import numpy as np
import matplotlib.pyplot as plt
import lab_07.measuring_predictions as mp
import gaussian_mixture_model as gmm

if __name__ == '__main__':
    data = np.loadtxt('../Project/trainData.txt', delimiter=",").T

    D, L = data[:data.shape[0] - 1, :], data[data.shape[0] - 1, :]

    (DTR, LTR), (DTE, LTE) = gmm.split_db_2to1(D, L)

    prior = 0.1
    C_fn = 1.0
    C_fp = 1.0
    working_point = (prior, C_fn, C_fp)

    covTypes = ['full', 'diagonal']
    n = [1, 2, 4, 8, 16, 32]

    for covType in covTypes:
        DCFs = []
        min_DCFs = []
        for numComponents in n:
            scores = gmm.GMM(DTR, LTR, DTE, numComponents, covType)
            L_pred = mp.optimal_bayes_decisions_bin(scores, working_point)
            err_rate = mp.err_rate(L_pred, LTE)
            confM = mp.confusion_matrix(L_pred, LTE)
            DCF = mp.bayes_risk_norm(working_point, confM)
            DCF_min = mp.min_DCF(scores, LTE, working_point)

            DCFs = np.append(DCFs, DCF)
            min_DCFs = np.append(min_DCFs, DCF_min)

        plt.figure()
        plt.title(f"{covType} DCF & minDCF vs numComponents")
        plt.plot(n, DCFs, '-o', label=f"DCF")
        plt.plot(n, min_DCFs, '-o', label=f"min DCF")
        plt.xlabel(f"numComponents")
        plt.legend()
        plt.show()