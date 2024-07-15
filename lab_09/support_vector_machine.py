import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.datasets import load_iris
import lab_07.measuring_predictions as mp

def vcol(vector):
    return vector.reshape(vector.size, 1)

def vrow(vector):
    return vector.reshape(1, vector.size)

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

def primal_formulation():
    #J_cap = 1.0/2.0 * np.linalg.norm(w_cap)**2 + C *    np.maximum(np.zeros())
    return 0


def kernel_func(D1, D2, c=None, d=None, gamma=None):
    if gamma is None:
        return (D1.T @ D2 + c) ** d
    else:
        D1Norms = (D1 ** 2).sum(0)
        D2Norms = (D2 ** 2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * D1.T @ D2
        return np.exp(-gamma * Z)


def dual_obj_wrap_bin(DTR, LTR, kernel=None, eps=None, c=None, d=None, gamma=None):
    def dual_obj_bin(alpha):
        Z = 2.0 * LTR - 1.0
        if kernel is None:
            G_cap = DTR.T @ DTR
        else:
            G_cap = kernel_func(DTR, DTR, c, d, gamma) + eps

        H_cap = vcol(Z) @ vrow(Z) * G_cap

        L = 0.5 * (vrow(alpha) @ H_cap @ vcol(alpha)).ravel() - vrow(alpha) @ np.ones(DTR.shape[1])

        L_grad = (H_cap @ vcol(alpha)).ravel() - 1.0

        return L, L_grad

    return dual_obj_bin

def Support_Vector_Machine(DTR, LTR, DTE, K=1, C=1.0):
    DTR_cap = np.vstack((DTR, np.ones(DTR.shape[1]) * K))
    DTE_cap = np.vstack((DTE, np.ones(DTE.shape[1]) * K))

    Z = 2.0 * LTR - 1.0

    dual_obj_bin = dual_obj_wrap_bin(DTR_cap, LTR)


    alpha, f_value, d = fmin_l_bfgs_b(dual_obj_bin, np.zeros((DTR_cap.shape[1])), bounds=np.array([(0, C) for i in range(DTR.shape[1])]), factr=1.0)


    w_cap_star = (vrow(alpha) * vrow(Z) * DTR_cap).sum(1)
    dual_scores = w_cap_star.T @ DTE_cap

    primal_solution = 0.5 * np.linalg.norm(w_cap_star) ** 2 + C * np.maximum(0, 1 - Z * (w_cap_star.T @ DTR_cap)).sum()
    dual_solution = -f_value


    print(f"primal_sol: {primal_solution}")
    print(f"dual_sol: {dual_solution}")
    dual_gap = primal_solution - dual_solution
    print(f"dual gap: {dual_gap}")
    return dual_scores


def Kernel_Support_Vector_Machine(DTR, LTR, DTE, C=1.0, eps=0.0, kernel=None, c=None, d=None, gamma=None):
    Z = 2.0 * LTR - 1.0

    dual_obj_bin = dual_obj_wrap_bin(DTR, LTR, kernel, eps, c, d, gamma)

    alpha, f_value, _ = fmin_l_bfgs_b(dual_obj_bin, np.zeros((DTR.shape[1])), bounds=np.array([(0, C) for i in range(DTR.shape[1])]), factr=1.0)

    dual_scores = (vcol(alpha * Z) * kernel_func(DTR, DTE, c, d, gamma)).sum(0)
    dual_solution = -f_value

    print(f"dual_sol: {dual_solution}")
    return dual_scores


if __name__ == '__main__':
    D, L = load_iris_binary()

    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    Ks = [1, 10]
    Cs = [0.1, 1.0, 10.0]
    prior = 0.5
    C_fn = 1.0
    C_fp = 1.0
    working_point = (prior, C_fn, C_fp)

    ##############################################
    ############## STANDARD SVM ##################
    ##############################################
    print("*"*50, "STANDARD", "*"*50, sep='\n')

    for K in Ks:
        for C in Cs:
            print(f"K={K} and C={C}: ")
            s = Support_Vector_Machine(DTR, LTR, DTE, K, C)
            L_pred = mp.optimal_bayes_decisions_bin(s, working_point)
            err_rate = mp.err_rate(L_pred, LTE)
            confM = mp.confusion_matrix(L_pred, LTE)
            DCF = mp.bayes_risk_norm(working_point, confM)
            DCF_min = mp.min_DCF(s, LTE, working_point)
            print(f"err_rate: {err_rate:.2f} % - DCF: {DCF:.4f}, min_DCF: {DCF_min:.4f}")

    ##############################################
    ############ KERNEL-POLY SVM #################
    ##############################################
    print("*" * 50, "KERNEL-POLY", "*" * 50, sep='\n')

    epss = [0.0, 1.0]
    cs = [0, 1]
    d = 2
    C = 1.0


    for eps in epss:
        for c in cs:
            print(f"C: {C} - eps: {eps} - c: {c} - d: {d}")
            s = Kernel_Support_Vector_Machine(DTR, LTR, DTE, C, eps, 'kernel', c, d)
            L_pred = mp.optimal_bayes_decisions_bin(s, working_point)
            err_rate = mp.err_rate(L_pred, LTE)
            confM = mp.confusion_matrix(L_pred, LTE)
            DCF = mp.bayes_risk_norm(working_point, confM)
            DCF_min = mp.min_DCF(s, LTE, working_point)
            print(f"err_rate: {err_rate:.2f} % - DCF: {DCF:.4f}, min_DCF: {DCF_min:.4f}")


    ##############################################
    ############# KERNEL-RBF SVM #################
    ##############################################
    print("*" * 50, "KERNEL-RBF", "*" * 50, sep='\n')

    gammas = [1.0, 10.0]

    for eps in epss:
        for gamma in gammas:
            print(f"C: {C} - eps: {eps} - gamma: {gamma}")
            s = Kernel_Support_Vector_Machine(DTR, LTR, DTE, C, eps, 'kernel', gamma=gamma)
            L_pred = mp.optimal_bayes_decisions_bin(s, working_point)
            err_rate = mp.err_rate(L_pred, LTE)
            confM = mp.confusion_matrix(L_pred, LTE)
            DCF = mp.bayes_risk_norm(working_point, confM)
            DCF_min = mp.min_DCF(s, LTE, working_point)
            print(f"err_rate: {err_rate:.2f} % - DCF: {DCF:.4f}, min_DCF: {DCF_min:.4f}")
