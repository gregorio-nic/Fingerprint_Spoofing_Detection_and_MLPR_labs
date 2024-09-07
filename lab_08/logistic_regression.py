import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.datasets import load_iris
import lab_07.measuring_predictions as mp

def vcol(vector):
    return vector.reshape(vector.size, 1)

def test_function(params):
    y = params[0]
    z = params[1]
    return (y+3) ** 2 + np.sin(y) + (z+1) ** 2

def load_iris_binary():
    D, L = load_iris()['data'].T, load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)#
    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE.astype(int))

def logreg_obj_wrap_bin(DTR, LTR, l):
    def logreg_obj_bin(v):
        w, b = v[:-1], v[-1]
        J = (l / 2.0) * (np.linalg.norm(w) ** 2)
        for i in range(DTR.shape[1]):
            J += (1.0 / DTR.shape[1]) * np.logaddexp(0.0, -(2.0 * float(LTR[i]) - 1.0) * (w.T @ DTR[:, i] + b))
        return J

    return logreg_obj_bin


def Logistic_Regression(DTR, LTR, DTE, l):
    n_0 = DTR[:, LTR == 0].shape[1]
    n_1 = DTR[:, LTR == 1].shape[1]
    logreg_obj_bin = logreg_obj_wrap_bin(DTR, LTR, l)
    x, f, d = fmin_l_bfgs_b(logreg_obj_bin, np.zeros((DTR.shape[0] + 1)), approx_grad=True)

    w = x[:DTR.shape[0]]
    b = x[DTR.shape[0]:]

    pi_emp = n_1 / (n_1 + n_0)

    scores = w.T @ DTE + b
    scores_llr = w.T @ DTE + b - np.log(pi_emp / (1.0 - pi_emp))
    return scores, scores_llr, f


def prior_weighted_logreg_obj_wrap_bin(DTR, LTR, l, pi, n_0, n_1):
    def logreg_obj_bin(v):
        w, b = v[:-1], v[-1]
        J = (l / 2.0) * (np.linalg.norm(w) ** 2)
        for i in range(DTR.shape[1]):
            if LTR[i] == 0:
                J += ((1.0-pi) / n_0) * np.logaddexp(0.0, -(2.0 * float(LTR[i]) - 1.0) * (w.T @ DTR[:, i] + b))
            else:
                J += (pi / n_1) * np.logaddexp(0.0, -(2.0 * float(LTR[i]) - 1.0) * (w.T @ DTR[:, i] + b))
        return J

    return logreg_obj_bin


def Prior_Weighted_Logistic_Regression(DTR, LTR, DTE, l, working_point):
    pi = working_point[0]
    n_0 = DTR[:, (LTR == 0)].shape[1]
    n_1 = DTR[:, (LTR == 1)].shape[1]
    prior_weighted_logreg_obj_bin = prior_weighted_logreg_obj_wrap_bin(DTR, LTR, l, pi, n_0, n_1)
    x, f, d = fmin_l_bfgs_b(prior_weighted_logreg_obj_bin, np.zeros((DTR.shape[0] + 1)), approx_grad=True)

    w = x[:DTR.shape[0]]
    b = x[DTR.shape[0]:]

    scores = w.T @ DTE + b
    scores_llr = w.T @ DTE + b - np.log(pi / (1.0 - pi))
    return scores, scores_llr, f

def quadratic_feature_expansion(X):
    X_T = X.T
    X_expanded = []
    for x in X_T:
        outer_product = np.outer(x, x).flatten()
        expanded_feature = np.concatenate([outer_product, x])
        X_expanded.append(expanded_feature)
    X_expanded = np.array(X_expanded).T

    return X_expanded

def Quadratic_Logistic_Regression(DTR, LTR, DTE, l):
    n_0 = DTR[:, LTR == 0].shape[1]
    n_1 = DTR[:, LTR == 1].shape[1]

    phi_DTR = quadratic_feature_expansion(DTR)

    logreg_obj_bin = logreg_obj_wrap_bin(phi_DTR, LTR, l)
    x, f, d = fmin_l_bfgs_b(logreg_obj_bin, np.zeros((phi_DTR.shape[0] + 1)), approx_grad=True)

    w = x[:phi_DTR.shape[0]]
    b = x[phi_DTR.shape[0]:]

    pi_emp = n_1 / (n_1 + n_0)

    phi_DTE = quadratic_feature_expansion(DTE)

    scores = w.T @ phi_DTE + b
    scores_llr = w.T @ phi_DTE + b - np.log(pi_emp / (1.0 - pi_emp))
    return scores, scores_llr, f

    

if __name__ == '__main__':

    ### TESTING NUMERICAL SOLVER
    x, f, d = fmin_l_bfgs_b(test_function, x0=np.array([0.0, 0.0]), approx_grad=True, iprint=1)

    print(f"x: {x}\nf: {f}\nd:{d}")

    ################################################
    ############## STANDARD LOG REG ################
    ################################################

    D, L = load_iris_binary()

    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)




    lambdas = [1.0e-3, 1.0e-1, 1.0]

    prior=0.5
    C_fn=1.0
    C_fp=1.0
    working_point = (prior, C_fn, C_fp)

    for l in lambdas:
        scores, scores_llr, f_value = Logistic_Regression(DTR, LTR, DTE, l)

        L_pred = mp.optimal_bayes_decisions_bin(scores, working_point)
        err_rate = mp.err_rate(L_pred, LTE)

        L_pred_llr = mp.optimal_bayes_decisions_bin(scores_llr, working_point)
        confM = mp.confusion_matrix(L_pred_llr, LTE)
        DCF = mp.bayes_risk_norm(working_point, confM)
        DCF_min = mp.min_DCF(scores_llr, LTE, working_point)

        print(f"Standard log_reg: working_point: {working_point} - lambda: {l} - f_value: {f_value} - error_rate: {err_rate} % - DCF: {DCF} - min_DCF:{DCF_min}")

    print("\n")
    ################################################
    ############ PRIOR WEIGHT LOG REG ##############
    ################################################

    prior = 0.8
    C_fn = 1.0
    C_fp = 1.0
    working_point = (prior, C_fn, C_fp)
    for l in lambdas:
        scores, scores_llr, f_value = Prior_Weighted_Logistic_Regression(DTR, LTR, DTE, l, working_point)

        L_pred = mp.optimal_bayes_decisions_bin(scores, working_point)
        err_rate = mp.err_rate(L_pred, LTE)

        L_pred_llr = mp.optimal_bayes_decisions_bin(scores_llr, working_point)
        confM = mp.confusion_matrix(L_pred_llr, LTE)
        DCF = mp.bayes_risk_norm(working_point, confM)
        DCF_min = mp.min_DCF(scores_llr, LTE, working_point)

        print(f"Prior-weighted log_reg: working_point: {working_point} - lambda: {l} - f_value: {f_value} - error_rate: {err_rate} % - DCF: {DCF} - min_DCF:{DCF_min}")