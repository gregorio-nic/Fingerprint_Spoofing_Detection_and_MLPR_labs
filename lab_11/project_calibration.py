import numpy as np
import sklearn
import matplotlib.pyplot as plt

import lab_08.logistic_regression as lr
import lab_07.measuring_predictions as mp
import lab_11.calibration as c

if __name__ == '__main__':
    data = np.loadtxt('../Project/trainData.txt', delimiter=",").T

    D, L = data[:data.shape[0] - 1, :], data[data.shape[0] - 1, :]

    (DTR, LTR), (DTE, LTE) = c.split_db_2to1(D, L)
    LTE = LTE.astype(int)

    prior = 0.1
    C_fn = 1.0
    C_fp = 1.0
    working_point = (prior, C_fn, C_fp)

    l = 0.1

    DCFs = []
    min_DCFs = []

    scores, scores_llr, f_value = lr.Quadratic_Logistic_Regression(DTR, LTR, DTE, l)

    L_pred = mp.optimal_bayes_decisions_bin(scores_llr, working_point)
    confM = mp.confusion_matrix(L_pred, LTE)
    DCF = mp.bayes_risk_norm(working_point, confM)
    DCF_min = mp.min_DCF(scores_llr, LTE, working_point)

    calibrated_scores_llr = c.single_fold_calibration(scores_llr, LTE, None, None, working_point, evaluation=False)

