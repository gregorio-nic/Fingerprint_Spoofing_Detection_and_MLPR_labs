import numpy as np
import sklearn
import matplotlib.pyplot as plt

import lab_08.logistic_regression as lr
import lab_07.measuring_predictions as mp
import lab_09.support_vector_machine as svm
import lab_10.gaussian_mixture_model as gmm
import lab_11.calibration as c

if __name__ == '__main__':
    data = np.loadtxt('../Project/trainData.txt', delimiter=",").T

    D, L = data[:data.shape[0] - 1, :], data[data.shape[0] - 1, :].astype(int)

    (DTR, LTR), (DTE, LTE) = c.split_db_2to1(D, L)
    LTE = LTE.astype(int)

    prior = 0.1
    C_fn = 1.0
    C_fp = 1.0
    working_point = (prior, C_fn, C_fp)

    model_scores = {}

    ##################################################
    ############# BEST LOGISTIC REGRESSION ###########
    ##################################################

    '''
    l = 0.1
    
    scores, scores_llr, f_value = lr.Quadratic_Logistic_Regression(DTR, LTR, DTE, l)

    model_scores['best_LR'] = scores_llr
    np.save('Data/best_LR', scores_llr)

    L_pred = mp.optimal_bayes_decisions_bin(scores_llr, working_point)
    confM = mp.confusion_matrix(L_pred, LTE)
    DCF = mp.bayes_risk_norm(working_point, confM)
    DCF_min = mp.min_DCF(scores_llr, LTE, working_point)

    calibrated_scores_llr = c.k_fold_calibration(scores_llr, LTE, None, None, working_point, KFOLD=5, evaluation=False)
    np.save('Data/calibrated_best_LR', calibrated_scores_llr)
    '''

    ##################################################
    ##################### BEST SVM ###################
    ##################################################

    '''
    eps = 1.0
    Cs = np.logspace(-3, 2, 11)
    C = Cs[9]
    gamma = np.exp(-2)

    scores = svm.Kernel_Support_Vector_Machine(DTR, LTR, DTE, C=C, kernel='RBF', gamma=gamma)

    model_scores['best_SVM'] = scores
    np.save('Data/best_SVM', scores)

    L_pred = mp.optimal_bayes_decisions_bin(scores, working_point)
    err_rate = mp.err_rate(L_pred, LTE)
    confM = mp.confusion_matrix(L_pred, LTE)
    DCF = mp.bayes_risk_norm(working_point, confM)
    DCF_min = mp.min_DCF(scores, LTE, working_point)

    calibrated_scores = c.k_fold_calibration(scores, LTE, None, None, working_point, KFOLD=5, evaluation=False)
    np.save('Data/calibrated_best_SVM', calibrated_scores)
    '''

    ##################################################
    ##################### BEST GMM ###################
    ##################################################

    '''
    covType = 'diagonal'
    numComponents = 4

    scores = gmm.GMM(DTR, LTR, DTE, numComponents, covType)

    model_scores['best_GMM'] = scores
    np.save('Data/best_GMM', scores)

    L_pred = mp.optimal_bayes_decisions_bin(scores, working_point)
    err_rate = mp.err_rate(L_pred, LTE)
    confM = mp.confusion_matrix(L_pred, LTE)
    DCF = mp.bayes_risk_norm(working_point, confM)
    DCF_min = mp.min_DCF(scores, LTE, working_point)

    calibrated_scores = c.k_fold_calibration(scores, LTE, None, None, working_point, KFOLD=5, evaluation=False)
    np.save('Data/calibrated_best_GMM', calibrated_scores)
    '''

    ######################################################
    ### COMPARING BEFORE AND AFTER CALIBRATION MODELS ####
    ######################################################




    ##################################################
    ###################### FUSION ####################
    ##################################################
    '''
    model_scores = {}
    lr_scores = np.load('Data/best_LR.npy')
    svm_scores = np.load('Data/best_SVM.npy')
    gmm_scores = np.load('Data/best_GMM.npy')

    fused_scores = {}
    fused_scores['LR + SVM'] = c.k_fold_fusion({'LR': lr_scores, 'SVM': svm_scores}, LTE, None, None, working_point, KFOLD=5, evaluation=False)
    fused_scores['LR + GMM'] = c.k_fold_fusion({'LR': lr_scores, 'GMM': gmm_scores}, LTE, None, None, working_point, KFOLD=5, evaluation=False)
    fused_scores['SVM + GMM'] = c.k_fold_fusion({'SVM': svm_scores, 'GMM': gmm_scores }, LTE, None, None, working_point, KFOLD=5, evaluation=False)
    fused_scores['LR + SVM + GMM'] = c.k_fold_fusion({'LR': lr_scores, 'SVM': svm_scores, 'GMM': gmm_scores }, LTE, None, None, working_point, KFOLD=5, evaluation=False)

    mp.bayes_error_plot(fused_scores, LTE, comparing=True)
    '''
    ##################################################
    #################### EVALUATION ##################
    ##################################################

    data = np.loadtxt('Data/evalData.txt', delimiter=",").T
    evalData, evalLabels = data[:data.shape[0] - 1, :], data[data.shape[0] - 1, :].astype(int)
    l = 0.1


    ######### FINAL LR TRAINING #########
    _, scores_llr, _ = lr.Quadratic_Logistic_Regression(D, L, D, l)

    #model_scores['best_LR'] = evalScoresLR
    #np.save('Data/best_LR', evalScoresLR)


    ######### FINAL GMM TRAINING #########
    covType = 'diagonal'
    numComponents = 4

    scores = gmm.GMM(D, L, D, numComponents, covType)

    #model_scores['best_GMM'] = scores
    #np.save('Data/best_GMM', scores)


    ######### FINAL FUSION TRAINING ##########
    fused_scores = c.k_fold_fusion({'LR': scores_llr, 'GMM': scores}, L, None, None, working_point, KFOLD=5, evaluation=False)

    ######### FINAL FINAL FINAL ##########

    _, evalScores_lr, _ = lr.Quadratic_Logistic_Regression(D, L, evalData, l)
    evalScores_gmm = gmm.GMM(D, L, evalData, numComponents, covType)



    ######### FINAL EVALUATION BEST SYSTEM ##########

    evalScores_fusion = c.k_fold_fusion({'LR': scores_llr, 'GMM': scores}, L, {'LR': evalScores_lr, 'GMM': evalScores_gmm}, evalLabels, working_point, KFOLD=5, evaluation=True)

    L_pred = mp.optimal_bayes_decisions_bin(evalScores_fusion, working_point)
    confM = mp.confusion_matrix(L_pred, evalLabels)
    DCF = mp.bayes_risk_norm(working_point, confM)
    DCF_min = mp.min_DCF(evalScores_fusion, evalLabels, working_point)

    mp.bayes_error_plot(evalScores_fusion, evalLabels)
