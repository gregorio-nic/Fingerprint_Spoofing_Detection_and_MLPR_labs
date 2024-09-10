import numpy as np
import matplotlib.pyplot as plt
import lab_07.measuring_predictions as mp
import lab_09.support_vector_machine as svm

if __name__ == '__main__':
    data = np.loadtxt('../Project/trainData.txt', delimiter=",").T

    D, L = data[:data.shape[0] - 1, :], data[data.shape[0] - 1, :]

    (DTR, LTR), (DTE, LTE) = svm.split_db_2to1(D, L)

    prior = 0.1
    C_fn = 1.0
    C_fp = 1.0
    working_point = (prior, C_fn, C_fp)

    ##############################################
    ############## STANDARD SVM ##################
    ##############################################
    print("*" * 50, "STANDARD", "*" * 50, sep='\n')

    K = 1.0
    Cs = np.logspace(-5, 0, 11)

    DCFs = []
    min_DCFs = []

    for C in Cs:
        print(f"K={K} and C={C}: ")
        s = svm.Support_Vector_Machine(DTR, LTR, DTE, K, C)
        L_pred = mp.optimal_bayes_decisions_bin(s, working_point)
        err_rate = mp.err_rate(L_pred, LTE)
        confM = mp.confusion_matrix(L_pred, LTE)
        DCF = mp.bayes_risk_norm(working_point, confM)
        DCF_min = mp.min_DCF(s, LTE, working_point)

        DCFs = np.append(DCFs, DCF)
        min_DCFs = np.append(min_DCFs, DCF_min)

    plt.figure()
    plt.title(f"STANDARD SVM DCF & minDCF vs C")
    plt.plot(Cs, DCFs, '-o', label=f"DCF")
    plt.plot(Cs, min_DCFs, '-o', label=f"min DCF")
    plt.xscale('log', base=10)
    plt.xlabel(f"C")
    plt.legend()
    plt.show()

    ##############################################
    ############ KERNEL-POLY SVM #################
    ##############################################
    print("*" * 50, "KERNEL-POLY", "*" * 50, sep='\n')

    eps = 0
    c = 1
    d = 2
    Cs = np.logspace(-5, 0, 11)

    DCFs = []
    min_DCFs = []

    for C in Cs:
        print(f"C: {C} - eps: {eps} - c: {c} - d: {d}")
        s = svm.Kernel_Support_Vector_Machine(DTR, LTR, DTE, C, eps, 'kernel', c, d)
        L_pred = mp.optimal_bayes_decisions_bin(s, working_point)
        err_rate = mp.err_rate(L_pred, LTE)
        confM = mp.confusion_matrix(L_pred, LTE)
        DCF = mp.bayes_risk_norm(working_point, confM)
        DCF_min = mp.min_DCF(s, LTE, working_point)

        DCFs = np.append(DCFs, DCF)
        min_DCFs = np.append(min_DCFs, DCF_min)

    plt.figure()
    plt.title(f"KERNEL POLYNOMIAL SVM DCF & minDCF vs C")
    plt.plot(Cs, DCFs, '-o', label=f"DCF")
    plt.plot(Cs, min_DCFs, '-o', label=f"min DCF")
    plt.xscale('log', base=10)
    plt.xlabel(f"C")
    plt.legend()
    plt.show()

    ##############################################
    ############ KERNEL-RBF SVM ##################
    ##############################################

    eps = 1
    Cs = np.logspace(-3, 2, 11)
    gammas = [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]

    DCFs = {
        gammas[0]: [],
        gammas[1]: [],
        gammas[2]: [],
        gammas[3]: []
    }
    min_DCFs = {
        gammas[0]: [],
        gammas[1]: [],
        gammas[2]: [],
        gammas[3]: []
    }

    for C in Cs:
        for gamma in gammas:
            print(f"C: {C} - eps: {eps} - gamma: {gamma}")
            s = svm.Kernel_Support_Vector_Machine(DTR, LTR, DTE, C, eps, 'kernel', gamma=gamma)
            L_pred = mp.optimal_bayes_decisions_bin(s, working_point)
            err_rate = mp.err_rate(L_pred, LTE)
            confM = mp.confusion_matrix(L_pred, LTE)
            DCF = mp.bayes_risk_norm(working_point, confM)
            DCF_min = mp.min_DCF(s, LTE, working_point)

            DCFs[gamma] = np.append(DCFs[gamma], DCF)
            min_DCFs[gamma] = np.append(min_DCFs[gamma], DCF_min)


    plt.figure()
    plt.title(f"KERNEL RBF SVM DCF & minDCF vs gamma vs C")
    for gamma in gammas:
        plt.plot(Cs, DCFs[gamma], '-o', label=f"DCF - gamma:{gamma: .2f}")
        plt.plot(Cs, min_DCFs[gamma], '-o', label=f"min DCF - gamma:{gamma: .2f}")
    plt.xscale('log', base=10)
    plt.xlabel(f"C")
    plt.legend()
    plt.show()
