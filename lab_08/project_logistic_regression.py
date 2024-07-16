import matplotlib.pyplot as plt

import lab_08.logistic_regression as lr
import lab_07.measuring_predictions as mp

import numpy as np


if __name__ == '__main__':
    data = np.loadtxt('../Project/trainData.txt', delimiter=",").T

    D, L = data[:data.shape[0] - 1, :], data[data.shape[0] - 1, :]

    (DTR, LTR), (DTE, LTE) = lr.split_db_2to1(D, L)

    lambdas = np.logspace(-4, 2, 13)

    prior = 0.1
    C_fn = 1.0
    C_fp = 1.0

    working_point = (prior, C_fn, C_fp)
    '''
    ##########################################################
    ###################### STANDARD LR #######################
    ##########################################################
    
    DCFs=[]
    min_DCFs=[]
    for l in lambdas:
        scores, scores_llr, f_value = lr.Logistic_Regression(DTR, LTR, DTE, l)

        L_pred = mp.optimal_bayes_decisions_bin(scores_llr, working_point)
        confM = mp.confusion_matrix(L_pred, LTE)
        DCF = mp.bayes_risk_norm(working_point, confM)
        DCF_min = mp.min_DCF(scores_llr, LTE, working_point)

        DCFs = np.append(DCFs, DCF)
        min_DCFs = np.append(min_DCFs, DCF_min)

    plt.figure()
    plt.title(f"STANDARD LR DCF & minDCF vs lambda")
    plt.plot(lambdas, DCFs, '-o', label=f"DCF")
    plt.plot(lambdas, min_DCFs, '-o', label=f"min DCF")
    plt.xscale('log', base=10)
    plt.xlabel(f"lambda")
    plt.legend()
    plt.show()



    ##########################################################
    ################# FEWER TRAINING SAMPLES #################
    ##########################################################

    DTR_filtered = DTR[:, ::50]
    LTR_filtered = LTR[::50]

    DCFs = []
    min_DCFs = []
    for l in lambdas:
        scores, scores_llr, f_value = lr.Logistic_Regression(DTR_filtered, LTR_filtered, DTE, l)

        L_pred = mp.optimal_bayes_decisions_bin(scores_llr, working_point)
        confM = mp.confusion_matrix(L_pred, LTE)
        DCF = mp.bayes_risk_norm(working_point, confM)
        DCF_min = mp.min_DCF(scores_llr, LTE, working_point)

        DCFs = np.append(DCFs, DCF)
        min_DCFs = np.append(min_DCFs, DCF_min)

    plt.figure()
    plt.title(f"SMALL DATASET LR DCF & minDCF vs lambda")
    plt.plot(lambdas, DCFs, '-o', label=f"DCF")
    plt.plot(lambdas, min_DCFs, '-o', label=f"min DCF")
    plt.xscale('log', base=10)
    plt.xlabel(f"lambda")
    plt.legend()
    plt.show()

    ##########################################################
    #################### PRIOR WEIGHT LR #####################
    ##########################################################

    DCFs = []
    min_DCFs = []
    for l in lambdas:
        scores, scores_llr, f_value = lr.Prior_Weighted_Logistic_Regression(DTR, LTR, DTE, l, working_point)

        L_pred = mp.optimal_bayes_decisions_bin(scores_llr, working_point)
        confM = mp.confusion_matrix(L_pred, LTE)
        DCF = mp.bayes_risk_norm(working_point, confM)
        DCF_min = mp.min_DCF(scores_llr, LTE, working_point)

        DCFs = np.append(DCFs, DCF)
        min_DCFs = np.append(min_DCFs, DCF_min)

    plt.figure()
    plt.title(f"PRIOR-WEIGHTED LR DCF & minDCF vs lambda")
    plt.plot(lambdas, DCFs, '-o', label=f"DCF")
    plt.plot(lambdas, min_DCFs, '-o', label=f"min DCF")
    plt.xscale('log', base=10)
    plt.xlabel(f"lambda")
    plt.legend()
    plt.show()

    ################################################
    ############# QUADRATIC LR ################
    ################################################

    DCFs = []
    min_DCFs = []
    for l in lambdas:
        scores, scores_llr, f_value = lr.Quadratic_Logistic_Regression(DTR, LTR, DTE, l)

        L_pred = mp.optimal_bayes_decisions_bin(scores_llr, working_point)
        confM = mp.confusion_matrix(L_pred, LTE)
        DCF = mp.bayes_risk_norm(working_point, confM)
        DCF_min = mp.min_DCF(scores_llr, LTE, working_point)

        DCFs = np.append(DCFs, DCF)
        min_DCFs = np.append(min_DCFs, DCF_min)

    plt.figure()
    plt.title(f"Quadratic LR DCF & minDCF vs lambda ")
    plt.plot(lambdas, DCFs, '-o', label=f"DCF")
    plt.plot(lambdas, min_DCFs, '-o', label=f"min DCF")
    plt.xscale('log', base=10)
    plt.xlabel(f"lambda ")
    plt.legend()
    plt.show()
    '''
    ################################################
    ############ CENTERING DATASET LR ##############
    ################################################

    DTR_mean = lr.vcol(np.mean(DTR, axis=1))
    DTR_centered = DTR-DTR_mean
    DTE_centered = DTE-DTR_mean

    DCFs = []
    min_DCFs = []
    for l in lambdas:
        scores, scores_llr, f_value = lr.Logistic_Regression(DTR_centered, LTR, DTE_centered, l, working_point)

        L_pred = mp.optimal_bayes_decisions_bin(scores_llr, working_point)
        confM = mp.confusion_matrix(L_pred, LTE)
        DCF = mp.bayes_risk_norm(working_point, confM)
        DCF_min = mp.min_DCF(scores_llr, LTE, working_point)

        DCFs = np.append(DCFs, DCF)
        min_DCFs = np.append(min_DCFs, DCF_min)

    plt.figure()
    plt.title(f"CENTERED DATASET - STANDARD LR DCF & minDCF vs lambda ")
    plt.plot(lambdas, DCFs, '-o', label=f"DCF")
    plt.plot(lambdas, min_DCFs, '-o', label=f"min DCF")
    plt.xscale('log', base=10)
    plt.xlabel(f"lambda ")
    plt.legend()
    plt.show()


