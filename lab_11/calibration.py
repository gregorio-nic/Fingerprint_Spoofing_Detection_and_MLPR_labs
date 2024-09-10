import numpy as np
import sklearn

import lab_08.logistic_regression as lr
import lab_07.measuring_predictions as mp


def vcol(vector):
    return vector.reshape(vector.size, 1)


def vrow(vector):
    return vector.reshape(1, vector.size)


def train_calibration_SINGLE_FOLD(DTR, LTR, DTE, working_point, l=0.0):
    scores, scores_llr, _ = lr.Prior_Weighted_Logistic_Regression(DTR, LTR, DTE, l, working_point)
    return scores_llr


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


def single_fold_calibration(scores, labels, eval_scores, eval_labels, working_point, evaluation=False):
    SCAL, SVAL = scores[::3], np.hstack([scores[1::3], scores[2::3]])
    LCAL, LVAL = labels[::3], np.hstack([labels[1::3], labels[2::3]])

    L_pred_val = mp.optimal_bayes_decisions_bin(SVAL, working_point)

    confM_val = mp.confusion_matrix(L_pred_val, LVAL)

    act_DCF_val = mp.bayes_risk_norm(working_point, confM_val)

    DCF_min_val = mp.min_DCF(SVAL, LVAL, working_point)

    print(f"BEFORE CALIBRATION (VALIDATION SET): min_DCF (pT={working_point[0]}) = {DCF_min_val},  actDCF_val (pT={working_point[0]}) = {act_DCF_val}")

    mp.bayes_error_plot(SVAL, LVAL)

    # Post-calibration (VALIDATION)
    SVAL_calibrated = train_calibration_SINGLE_FOLD(vrow(SCAL), LCAL, vrow(SVAL), working_point, l=0.0)

    minDCF_val = mp.min_DCF(SVAL_calibrated, LVAL, working_point)
    L_pred_val = mp.optimal_bayes_decisions_bin(SVAL_calibrated, working_point)
    confM_val = mp.confusion_matrix(L_pred_val, LVAL)

    actDCF_val = mp.bayes_risk_norm(working_point, confM_val)
    print(f'AFTER CALIBRATION (VALIDATION SET): minDCF (pT={working_point[0]}) = {minDCF_val} - actDCF (pT={working_point[0]}) = {actDCF_val}')
    mp.bayes_error_plot(SVAL_calibrated, LVAL)

    if evaluation is True:
        # Post-calibration (EVALUATION)
        eval_scores_calibrated = train_calibration_SINGLE_FOLD(vrow(SCAL), LCAL, vrow(eval_scores), working_point, l=0.0)

        minDCF_eval = mp.min_DCF(eval_scores_calibrated, eval_labels, working_point)
        L_pred_eval = mp.optimal_bayes_decisions_bin(eval_scores_calibrated, working_point)
        confM_eval = mp.confusion_matrix(L_pred_eval, eval_labels)

        actDCF_eval = mp.bayes_risk_norm(working_point, confM_eval)
        print(f'AFTER CALIBRATION (EVALUATION SET): minDCF (pT={working_point[0]}) = {minDCF_eval} - actDCF (pT={working_point[0]}) = {actDCF_eval}')
        return eval_scores_calibrated
    return SVAL_calibrated

def k_fold_calibration(scores, labels, eval_scores, eval_labels, working_point, KFOLD, evaluation=False):
    size = int(labels.size / KFOLD)

    cal_models = {}

    calibrated_scores_val = np.array([])
    calibrated_labels_val = np.array([])

    for i in range(KFOLD):
        # Building folds for each calibration model
        cal_models[i] = (np.hstack((scores[:(size * i)], scores[(size * (i + 1)):])),
                         np.hstack((labels[:(size * i)], labels[(size * (i + 1)):])),
                         scores[(size * i):(size * (i + 1))], labels[(size * i):(size * (i + 1))])

        # Post-calibration
        calibrated_scores_val = np.hstack((calibrated_scores_val,
                                         train_calibration_SINGLE_FOLD(vrow(cal_models[i][0]), cal_models[i][1],
                                                                       vrow(cal_models[i][2]), working_point, l=0.0)))
        calibrated_labels_val = np.hstack((calibrated_labels_val, cal_models[i][3]))


    L_pred_val = mp.optimal_bayes_decisions_bin(scores, working_point)

    confM_val = mp.confusion_matrix(L_pred_val, labels)

    act_DCF_val = mp.bayes_risk_norm(working_point, confM_val)

    DCF_min_val = mp.min_DCF(scores, labels, working_point)

    print(f"BEFORE CALIBRATION (VALIDATION SET): min_DCF (pT={working_point[0]}) = {DCF_min_val},  actDCF_val (pT={working_point[0]}) = {act_DCF_val}")

    calibrated_labels_val = calibrated_labels_val.astype(int)
    minDCF_val = mp.min_DCF(calibrated_scores_val, calibrated_labels_val, working_point)
    predicted_labels_val = mp.optimal_bayes_decisions_bin(calibrated_scores_val, working_point)
    confM_val = mp.confusion_matrix(predicted_labels_val, calibrated_labels_val)

    actDCF_val = mp.bayes_risk_norm(working_point, confM_val)
    print(f'AFTER CALIBRATION (VALIDATION SET): minDCF (pT={working_point[0]}) = {minDCF_val} - actDCF (pT={working_point[0]}) = {actDCF_val}')

    mp.bayes_error_plot(scores, labels)
    mp.bayes_error_plot(calibrated_scores_val, calibrated_labels_val)

    if evaluation is True:
        # Post-calibration (EVALUATION)
        eval_scores_calibrated = train_calibration_SINGLE_FOLD(vrow(scores), labels, vrow(eval_scores), working_point, l=0.0)

        minDCF_eval = mp.min_DCF(eval_scores_calibrated, eval_labels, working_point)
        L_pred_eval = mp.optimal_bayes_decisions_bin(eval_scores_calibrated, working_point)
        confM_eval = mp.confusion_matrix(L_pred_eval, eval_labels)

        actDCF_eval = mp.bayes_risk_norm(working_point, confM_eval)
        print(f'AFTER CALIBRATION (EVALUATION SET): minDCF (pT={working_point[0]}) = {minDCF_eval} - actDCF (pT={working_point[0]}) = {actDCF_eval}')
        return eval_scores_calibrated
    return calibrated_scores_val

def single_fold_fusion(scores: dict, labels, eval_scores: dict, eval_labels, working_point, evaluation=False):

    stacked_scores_cal = np.array([])
    stacked_scores_val = np.array([])
    stacked_scores_eval = np.array([])
    for score in scores:
        SCAL, SVAL = score[::3], np.hstack([score[1::3], score[2::3]])
        stacked_scores_cal = np.vstack([stacked_scores_cal, SCAL])
        stacked_scores_val = np.vstack([stacked_scores_val, SVAL])

    if evaluation is True:
        for eval_score in eval_scores:
            stacked_scores_eval = np.vstack([stacked_scores_eval, eval_score])

    LCAL, LVAL = labels[::3], np.hstack([labels[1::3], labels[2::3]])

    fused_SVAL = train_calibration_SINGLE_FOLD(stacked_scores_cal, LCAL, stacked_scores_val,
                                               working_point, l=0.0)

    L_pred_VAL = mp.optimal_bayes_decisions_bin(fused_SVAL, working_point)

    confM_VAL = mp.confusion_matrix(L_pred_VAL, LVAL)

    act_DCF_VAL = mp.bayes_risk_norm(working_point, confM_VAL)

    DCF_min_VAL = mp.min_DCF(fused_SVAL, LVAL, working_point)

    print(f"AFTER FUSION (VALIDATION SET): min_DCF (pT={working_point[0]}) = {DCF_min_VAL}, actDCF_val (pT={working_point[0]}) = {act_DCF_VAL}")

    mp.bayes_error_plot(fused_SVAL, LVAL)

    if evaluation is True:
        # Post-calibration
        fused_EVAL = train_calibration_SINGLE_FOLD(stacked_scores_cal, LCAL, stacked_scores_eval,
                                                   working_point, l=0.0)

        minDCF_EVAL = mp.min_DCF(fused_EVAL, eval_labels, working_point)
        L_pred_EVAL = mp.optimal_bayes_decisions_bin(fused_EVAL, working_point)
        confM_EVAL = mp.confusion_matrix(L_pred_EVAL, eval_labels)

        actDCF_EVAL = mp.bayes_risk_norm(working_point, confM_EVAL)
        print(f'AFTER FUSION (EVALUATION SET): minDCF (pT={working_point[0]}) = {minDCF_EVAL} - actDCF (pT={working_point[0]}) = {actDCF_EVAL}')

        mp.bayes_error_plot(fused_EVAL, eval_labels)
        return fused_EVAL
    return fused_SVAL

def k_fold_fusion(scores: dict, labels, eval_scores: dict, eval_labels, working_point, KFOLD, evaluation=False):
    size = int(labels.size / KFOLD)

    cal_models = {}

    fused_scores_VAL = np.array([])
    fused_labels_VAL = np.array([])
    for i in range(KFOLD):
        SCAL = np.array([])
        SVAL = np.array([])
        for score in scores.values():
            if SCAL.size == 0:
                SCAL = np.hstack((score[:(size * i)], score[(size * (i + 1)):]))
            else:
                SCAL = np.vstack([SCAL, np.hstack((score[:(size * i)], score[(size * (i + 1)):]))])
            if SVAL.size == 0:
                SVAL = score[(size * i):(size * (i + 1))]
            else:
                SVAL = np.vstack([SVAL, score[(size * i):(size * (i + 1))]])

        # Building folds for each calibration model
        cal_models[i] = (
            SCAL,
            np.hstack((labels[:(size * i)], labels[(size * (i + 1)):])),
            SVAL,
            labels[(size * i):(size * (i + 1))]
        )

        # Post-calibration
        fused_scores_VAL = np.hstack((fused_scores_VAL,
                                      train_calibration_SINGLE_FOLD(cal_models[i][0],
                                                                    cal_models[i][1],
                                                                    cal_models[i][2],
                                                                    working_point,
                                                                    l=0.0)
                                      ))

        fused_labels_VAL = np.hstack((fused_labels_VAL, cal_models[i][3]))

    fused_labels_VAL = fused_labels_VAL.astype(int)
    minDCF_VAL = mp.min_DCF(fused_scores_VAL, fused_labels_VAL, working_point)
    predicted_labels_VAL = mp.optimal_bayes_decisions_bin(fused_scores_VAL, working_point)
    confM_VAL = mp.confusion_matrix(predicted_labels_VAL, fused_labels_VAL)

    actDCF_VAL = mp.bayes_risk_norm(working_point, confM_VAL)
    print(f'AFTER FUSION (VALIDATION SET): minDCF (pT={working_point[0]}) = {minDCF_VAL} - actDCF (pT={working_point[0]}) = {actDCF_VAL}')

    mp.bayes_error_plot(fused_scores_VAL, fused_labels_VAL)

    # EVALUATION
    if evaluation is True:
        scores_CAL = np.array([])
        for score in scores.values():
            if scores_CAL.size == 0:
                scores_CAL = score
            else:
                scores_CAL = np.vstack([scores_CAL, score])


        scores_EVAL = np.array([])
        for score in eval_scores.values():
            if scores_EVAL.size == 0:
                scores_EVAL = score
            else:
                scores_EVAL = np.vstack([scores_EVAL, score])


        fused_scores_EVAL = train_calibration_SINGLE_FOLD(scores_CAL,
                                                          labels,
                                                          scores_EVAL,
                                                          working_point,
                                                          l=0.0
                                                          )


        minDCF_EVAL = mp.min_DCF(fused_scores_EVAL, eval_labels, working_point)
        predicted_labels_EVAL = mp.optimal_bayes_decisions_bin(fused_scores_EVAL, working_point)
        confM_EVAL = mp.confusion_matrix(predicted_labels_EVAL, eval_labels)

        actDCF_EVAL = mp.bayes_risk_norm(working_point, confM_EVAL)
        print(f'AFTER FUSION (EVALUATION SET): minDCF (pT={working_point[0]}) = {minDCF_EVAL} - actDCF (pT={working_point[0]}) = {actDCF_EVAL}')

        mp.bayes_error_plot(fused_scores_EVAL, eval_labels)

        # Used to compare system1 vs system 2 vs fusion dcfs
        eval_scores['Fusion'] = fused_scores_EVAL

        mp.bayes_error_plot(eval_scores, eval_labels, comparing=True)
        return fused_scores_EVAL
    return fused_scores_VAL


if __name__ == '__main__':
    '''
    scores_1 = np.load('Data/scores_1.npy')
    scores_2 = np.load('Data/scores_2.npy')

    labels = np.load('Data/labels.npy')

    prior = 0.2
    C_fn = 1.0
    C_fp = 1.0
    working_point = (prior, C_fn, C_fp)

    scores = {
        'system1': scores_1,
        'system2': scores_2,
    }

    s1_L_pred = mp.optimal_bayes_decisions_bin(scores_1, working_point)
    s2_L_pred = mp.optimal_bayes_decisions_bin(scores_2, working_point)

    s1_confM = mp.confusion_matrix(s1_L_pred, labels)
    s2_confM = mp.confusion_matrix(s2_L_pred, labels)

    
    s1_act_DCF = mp.bayes_risk_norm(working_point, s1_confM)
    s2_act_DCF = mp.bayes_risk_norm(working_point, s2_confM)

    s1_DCF_min = mp.min_DCF(scores_1, labels, working_point)
    s2_DCF_min = mp.min_DCF(scores_2, labels, working_point)

    print(f"system 1: actDCF= {s1_act_DCF}, min_DCF={s1_DCF_min}")
    print(f"system 2: actDCF= {s2_act_DCF}, min_DCF={s2_DCF_min}")

    mp.bayes_error_plot(scores, labels, comparing=True)
    
    
    ############################################################
    ################# SINGLE-FOLD CALIBRATION ##################
    ############################################################

    SCAL1, SVAL1 = scores_1[::3], np.hstack([scores_1[1::3], scores_1[2::3]])
    SCAL2, SVAL2 = scores_2[::3], np.hstack([scores_2[1::3], scores_2[2::3]])

    LCAL, LVAL = labels[::3], np.hstack([labels[1::3], labels[2::3]])

    s1_L_pred_cal = mp.optimal_bayes_decisions_bin(SCAL1, working_point)
    s1_L_pred_val = mp.optimal_bayes_decisions_bin(SVAL1, working_point)

    s2_L_pred_cal = mp.optimal_bayes_decisions_bin(SCAL2, working_point)
    s2_L_pred_val = mp.optimal_bayes_decisions_bin(SVAL2, working_point)

    s1_confM_cal = mp.confusion_matrix(s1_L_pred_cal, LCAL)
    s1_confM_val = mp.confusion_matrix(s1_L_pred_val, LVAL)

    s2_confM_cal = mp.confusion_matrix(s2_L_pred_cal, LCAL)
    s2_confM_val = mp.confusion_matrix(s2_L_pred_val, LVAL)

    s1_act_DCF_cal = mp.bayes_risk_norm(working_point, s1_confM_cal)
    s1_act_DCF_val = mp.bayes_risk_norm(working_point, s1_confM_val)

    s2_act_DCF_cal = mp.bayes_risk_norm(working_point, s2_confM_cal)
    s2_act_DCF_val = mp.bayes_risk_norm(working_point, s2_confM_val)

    s1_DCF_min = mp.min_DCF(scores_1, labels, working_point)
    s2_DCF_min = mp.min_DCF(scores_2, labels, working_point)

    #print(f"system 1: actDCF_cal= {s1_act_DCF_cal}, min_DCF={s1_DCF_min}")
    print(f"system 1: actDCF_val= {s1_act_DCF_val}, min_DCF={s1_DCF_min}")

    #print(f"system 2: actDCF_cal= {s2_act_DCF_cal}, min_DCF={s2_DCF_min}")
    print(f"system 2: actDCF_val= {s2_act_DCF_val}, min_DCF={s2_DCF_min}")

    # mp.bayes_error_plot(SCAL1, LCAL) VA FATTO SOLO SU VAL
    mp.bayes_error_plot(SVAL1, LVAL)

    # mp.bayes_error_plot(SCAL2, LCAL) VA FATTO SOLO SU VAL
    mp.bayes_error_plot(SVAL2, LVAL)

    # Post-calibration
    cal_score_1 = train_calibration_SINGLE_FOLD(vrow(SCAL1), LCAL, vrow(SVAL1), working_point, l=0.0)

    minDCF = mp.min_DCF(cal_score_1, LVAL, working_point)
    L_pred_1 = mp.optimal_bayes_decisions_bin(cal_score_1, working_point)
    confM = mp.confusion_matrix(L_pred_1, LVAL)

    actDCF = mp.bayes_risk_norm(working_point, confM)
    print(f'Sys1: minDCF (0.2) = {minDCF} - actDCF (0.2) = {actDCF}')

    mp.bayes_error_plot(cal_score_1, LVAL)




    cal_score_2 = train_calibration_SINGLE_FOLD(vrow(SCAL2), LCAL, vrow(SVAL2), working_point, l=0.0)

    minDCF = mp.min_DCF(cal_score_2, LVAL, working_point)
    L_pred_2 = mp.optimal_bayes_decisions_bin(cal_score_2, working_point)
    confM = mp.confusion_matrix(L_pred_2, LVAL)

    actDCF = mp.bayes_risk_norm(working_point, confM)
    print(f'Sys2: minDCF (0.2) = {minDCF} - actDCF (0.2) = {actDCF}')

    mp.bayes_error_plot(cal_score_2, LVAL)
    ############################################################
    #################### K-FOLD CALIBRATION ####################
    ############################################################

    KFOLD = 5
    size = int(labels.size / KFOLD)

    # System 1
    cal_models = {}

    calibrated_scores_1 = np.array([])
    calibrated_labels_1 = np.array([])

    for i in range(KFOLD):
        # Building folds for each calibration model
        cal_models[i] = (np.hstack((scores_1[:(size*i)], scores_1[(size*(i+1)):])), np.hstack((labels[:(size*i)], labels[(size*(i+1)):])), scores_1[(size * i):(size * (i + 1))], labels[(size * i):(size * (i + 1))])

        # Post-calibration
        calibrated_scores_1 = np.hstack((calibrated_scores_1, train_calibration_SINGLE_FOLD(vrow(cal_models[i][0]), cal_models[i][1], vrow(cal_models[i][2]), working_point, l=0.0)))
        calibrated_labels_1 = np.hstack((calibrated_labels_1, cal_models[i][3]))

    calibrated_labels_1 = calibrated_labels_1.astype(int)
    minDCF = mp.min_DCF(calibrated_scores_1, calibrated_labels_1, working_point)
    predicted_labels_1 = mp.optimal_bayes_decisions_bin(calibrated_scores_1, working_point)
    confM = mp.confusion_matrix(predicted_labels_1, calibrated_labels_1)

    actDCF = mp.bayes_risk_norm(working_point, confM)
    print(f'Sys1: minDCF (0.2) = {minDCF} - actDCF (0.2) = {actDCF}')



    mp.bayes_error_plot(scores_1, labels)
    mp.bayes_error_plot(calibrated_scores_1, calibrated_labels_1)

    # System 2
    cal_models = {}

    calibrated_scores_2 = np.array([])
    calibrated_labels_2 = np.array([])

    for i in range(KFOLD):
        # Building folds for each calibration model
        cal_models[i] = (np.hstack((scores_2[:(size * i)], scores_2[(size * (i + 1)):])),
                         np.hstack((labels[:(size * i)], labels[(size * (i + 1)):])),
                         scores_1[(size * i):(size * (i + 1))], labels[(size * i):(size * (i + 1))])

        # Post-calibration
        calibrated_scores_2 = np.hstack((calibrated_scores_2,
                                       train_calibration_SINGLE_FOLD(vrow(cal_models[i][0]), cal_models[i][1],
                                                                     vrow(cal_models[i][2]), working_point, l=0.0)))
        calibrated_labels_2 = np.hstack((calibrated_labels_2, cal_models[i][3]))

    calibrated_labels_2 = calibrated_labels_2.astype(int)
    minDCF = mp.min_DCF(calibrated_scores_2, calibrated_labels_2, working_point)
    predicted_labels_2 = mp.optimal_bayes_decisions_bin(calibrated_scores_2, working_point)
    confM = mp.confusion_matrix(predicted_labels_2, calibrated_labels_2)

    actDCF = mp.bayes_risk_norm(working_point, confM)
    print(f'Sys1: minDCF (0.2) = {minDCF} - actDCF (0.2) = {actDCF}')

    mp.bayes_error_plot(scores_2, labels)
    mp.bayes_error_plot(calibrated_scores_2, calibrated_labels_2)

    ############################################################
    #################### SINGLE-FOLD FUSION ####################
    ############################################################

    SCAL1, SVAL1 = scores_1[::3], np.hstack([scores_1[1::3], scores_1[2::3]])
    SCAL2, SVAL2 = scores_2[::3], np.hstack([scores_2[1::3], scores_2[2::3]])

    LCAL, LVAL = labels[::3], np.hstack([labels[1::3], labels[2::3]])

    SEVAL1, SEVAL2 = np.load('Data/eval_scores_1.npy'), np.load('Data/eval_scores_2.npy')
    LEVAL = np.load('Data/eval_labels.npy').astype(int)

    fused_SVAL = train_calibration_SINGLE_FOLD(np.vstack([SCAL1, SCAL2]), LCAL, np.vstack([SVAL1, SVAL2]), working_point, l=0.0)


    L_pred_VAL = mp.optimal_bayes_decisions_bin(fused_SVAL, working_point)

    confM_VAL = mp.confusion_matrix(L_pred_VAL, LVAL)

    act_DCF_VAL = mp.bayes_risk_norm(working_point, confM_VAL)

    DCF_min_VAL = mp.min_DCF(fused_SVAL, LVAL, working_point)


    print(f"FUSION VALIDATION: actDCF_val= {act_DCF_VAL}, min_DCF={DCF_min_VAL}")

    mp.bayes_error_plot(fused_SVAL, LVAL)

    # Post-calibration
    fused_EVAL = train_calibration_SINGLE_FOLD(np.vstack([SCAL1, SCAL2]), LCAL, np.vstack([SEVAL1, SEVAL2]), working_point, l=0.0)

    minDCF_EVAL = mp.min_DCF(fused_EVAL, LEVAL, working_point)
    L_pred_EVAL = mp.optimal_bayes_decisions_bin(fused_EVAL, working_point)
    confM_EVAL = mp.confusion_matrix(L_pred_EVAL, LEVAL)

    actDCF_EVAL = mp.bayes_risk_norm(working_point, confM_EVAL)
    print(f'FUSION EVALUATION: minDCF (0.2) = {minDCF_EVAL} - actDCF (0.2) = {actDCF_EVAL}')

    mp.bayes_error_plot(fused_EVAL, LEVAL)

    
    ############################################################
    ###################### K-FOLD FUSION #######################
    ############################################################

    KFOLD = 5
    size = int(labels.size / KFOLD)

    cal_models_1 = {}
    cal_models_2 = {}

    fused_scores_VAL = np.array([])
    fused_labels_VAL = np.array([])
    for i in range(KFOLD):
        # Building folds for each calibration model
        cal_models_1[i] = (np.hstack((scores_1[:(size * i)], scores_1[(size * (i + 1)):])),
                           np.hstack((labels[:(size * i)], labels[(size * (i + 1)):])),
                           scores_1[(size * i):(size * (i + 1))], labels[(size * i):(size * (i + 1))])

        cal_models_2[i] = (np.hstack((scores_2[:(size * i)], scores_2[(size * (i + 1)):])),
                           np.hstack((labels[:(size * i)], labels[(size * (i + 1)):])),
                           scores_2[(size * i):(size * (i + 1))], labels[(size * i):(size * (i + 1))])

        # Post-calibration
        fused_scores_VAL = np.hstack((fused_scores_VAL,
                                      train_calibration_SINGLE_FOLD(np.vstack([cal_models_1[i][0], cal_models_2[i][0]]),
                                                                    cal_models_1[i][1],
                                                                    np.vstack([cal_models_1[i][2], cal_models_2[i][2]]),
                                                                    working_point,
                                                                    l=0.0)
                                      ))

        fused_labels_VAL = np.hstack((fused_labels_VAL, cal_models_1[i][3]))

    fused_labels_VAL = fused_labels_VAL.astype(int)
    minDCF = mp.min_DCF(fused_scores_VAL, fused_labels_VAL, working_point)
    predicted_labels_VAL = mp.optimal_bayes_decisions_bin(fused_scores_VAL, working_point)
    confM = mp.confusion_matrix(predicted_labels_VAL, fused_labels_VAL)

    actDCF = mp.bayes_risk_norm(working_point, confM)
    print(f'Fusion: minDCF (0.2) = {minDCF} - actDCF (0.2) = {actDCF}')

    mp.bayes_error_plot(fused_scores_VAL, fused_labels_VAL)

    # Used to compare system1 vs system 2 vs fusion dcfs
    #comparing = {
    #    'system1':scores_1,
    #    'system2':scores_2,
    #    'fusion':fused_scores_VAL
    #}
    #mp.bayes_error_plot(comparing, fused_labels_VAL, comparing=True)

    # EVALUATION

    SEVAL1, SEVAL2 = np.load('Data/eval_scores_1.npy'), np.load('Data/eval_scores_2.npy')
    LEVAL = np.load('Data/eval_labels.npy').astype(int)

    cal_models_1 = {}
    cal_models_2 = {}

    fused_scores_EVAL = np.array([])

    fused_scores_EVAL = np.hstack((fused_scores_EVAL,
                                   train_calibration_SINGLE_FOLD(np.vstack([scores_1, scores_2]),
                                                                 labels,
                                                                 np.vstack([SEVAL1, SEVAL2]),
                                                                 working_point,
                                                                 l=0.0)
                                   ))

    fused_labels_EVAL = LEVAL

    fused_labels_EVAL = fused_labels_EVAL.astype(int)
    minDCF = mp.min_DCF(fused_scores_EVAL, fused_labels_EVAL, working_point)
    predicted_labels_EVAL = mp.optimal_bayes_decisions_bin(fused_scores_EVAL, working_point)
    confM = mp.confusion_matrix(predicted_labels_EVAL, fused_labels_EVAL)

    actDCF = mp.bayes_risk_norm(working_point, confM)
    print(f'Fusion: minDCF (0.2) = {minDCF} - actDCF (0.2) = {actDCF}')

    mp.bayes_error_plot(fused_scores_EVAL, fused_labels_EVAL)

    # Used to compare system1 vs system 2 vs fusion dcfs
    comparing = {
        'system1': SEVAL1,
        'system2': SEVAL2,
        'fusion': fused_scores_EVAL
    }
    mp.bayes_error_plot(comparing, fused_labels_EVAL, comparing=True)
    '''