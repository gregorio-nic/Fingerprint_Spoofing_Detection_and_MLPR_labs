import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import sklearn.datasets

'''turns vector into row-matrix'''


def vrow(vector):
    return vector.reshape(1, vector.size)


'''turns vector into col-matrix'''


def vcol(vector):
    return vector.reshape(vector.size, 1)


def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    LTR = L[idxTrain]

    DTE = D[:, idxTest]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)


def mean_cov_iris(DTR, LTR, bin=False):
    if not bin:
        L_0 = (LTR == 0)
    L_1 = (LTR == 1)
    L_2 = (LTR == 2)
    if not bin:
        m_0 = vcol(np.mean(DTR[:, L_0], axis=1))
    m_1 = vcol(np.mean(DTR[:, L_1], axis=1))
    m_2 = vcol(np.mean(DTR[:, L_2], axis=1))

    if not bin:
        c_0 = 1 / (DTR[:, L_0].shape[1]) * (DTR[:, L_0] - m_0) @ (DTR[:, L_0] - m_0).T
    c_1 = 1 / (DTR[:, L_1].shape[1]) * (DTR[:, L_1] - m_1) @ (DTR[:, L_1] - m_1).T
    c_2 = 1 / (DTR[:, L_2].shape[1]) * (DTR[:, L_2] - m_2) @ (DTR[:, L_2] - m_2).T
    if not bin:
        return (m_0, c_0), (m_1, c_1), (m_2, c_2)
    else:
        return (m_1, c_1), (m_2, c_2)


def loglikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum(0)


'''log density for a data matrix X[M, N]'''


def logpdf_GAU_ND(X, mu, C):
    Y = np.array([logpdf_GAU_ND_1_sample(X[:, i:i + 1], mu, C) for i in range(X.shape[1])]).ravel()
    return Y


'''log density for single sample x=np.array([M, 1])'''


def logpdf_GAU_ND_1_sample(x, mu, C):
    M = mu.shape[0]
    return - (M / 2) * np.log(2 * np.pi) - (1 / 2) * np.linalg.slogdet(C)[1] - (1 / 2) * (x - mu).T @ np.linalg.inv(
        C) @ (x - mu)


def err_rate(L_pred, LTE):
    return np.count_nonzero(L_pred - LTE) / LTE.shape[0] * 100

def tied_cov(DTR, LTR, bin=False):
    if not bin:
        L_0 = (LTR == 0)
    L_1 = (LTR == 1)
    L_2 = (LTR == 2)

    if not bin:
        m_0 = vcol(np.mean(DTR[:, L_0], axis=1))
    m_1 = vcol(np.mean(DTR[:, L_1], axis=1))
    m_2 = vcol(np.mean(DTR[:, L_2], axis=1))

    if not bin:
        c_0 = (DTR[:, L_0] - m_0) @ (DTR[:, L_0] - m_0).T
    c_1 = (DTR[:, L_1] - m_1) @ (DTR[:, L_1] - m_1).T
    c_2 = (DTR[:, L_2] - m_2) @ (DTR[:, L_2] - m_2).T

    if not bin:
        c = c_0 + c_1 + c_2
    else:
        c = c_1 + c_2
    c = 1/(DTR.shape[1]) * c

    return c



if __name__ == '__main__':
    pi = np.ones((3, 1)) * 1 / 3
    print(pi)
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    ###################################################
    #################------STANDARD------##############
    ###################################################
    (m_0, c_0), (m_1, c_1), (m_2, c_2) = mean_cov_iris(DTR, LTR)

    S = np.exp(logpdf_GAU_ND(DTE, m_0, c_0))
    S = np.vstack((S, np.exp(logpdf_GAU_ND(DTE, m_1, c_1))))
    S = np.vstack((S, np.exp(logpdf_GAU_ND(DTE, m_2, c_2))))

    SJoint = S * pi

    SMarginal = vrow(SJoint.sum(0))

    SPost = SJoint / SMarginal

    L_pred = np.argmax(SPost, 0)
    print(err_rate(L_pred, LTE), "%")

    ###################################################
    ######-----STANDARD w/ log_densities & scores----###
    ###################################################

    log_S = logpdf_GAU_ND(DTE, m_0, c_0)
    log_S = np.vstack((log_S, logpdf_GAU_ND(DTE, m_1, c_1)))
    log_S = np.vstack((log_S, logpdf_GAU_ND(DTE, m_2, c_2)))

    log_SJoint = log_S + np.log(pi)

    log_SMarginal = vrow(scipy.special.logsumexp(log_SJoint, axis=0))

    log_SPost = log_SJoint - log_SMarginal

    exp_log_SPost = np.exp(log_SPost)

    ###################################################
    #################-----NAIVE-BAYES----##############
    ###################################################

    identity = np.eye(DTR.shape[0])

    nb_c0 = c_0 * identity
    nb_c1 = c_1 * identity
    nb_c2 = c_2 * identity

    S = np.exp(logpdf_GAU_ND(DTE, m_0, nb_c0))
    S = np.vstack((S, np.exp(logpdf_GAU_ND(DTE, m_1, nb_c1))))
    S = np.vstack((S, np.exp(logpdf_GAU_ND(DTE, m_2, nb_c2))))

    SJoint = S * pi

    SMarginal = vrow(SJoint.sum(0))

    SPost = SJoint / SMarginal

    L_pred = np.argmax(SPost, 0)
    print(err_rate(L_pred, LTE), "%")

    ###################################################
    ####################-----TIED----##################
    ###################################################

    c = tied_cov(DTR, LTR)

    print("******"*50, c)

    S = np.exp(logpdf_GAU_ND(DTE, m_0, c))
    S = np.vstack((S, np.exp(logpdf_GAU_ND(DTE, m_1, c))))
    S = np.vstack((S, np.exp(logpdf_GAU_ND(DTE, m_2, c))))

    SJoint = S * pi

    SMarginal = vrow(SJoint.sum(0))

    SPost = SJoint / SMarginal

    L_pred = np.argmax(SPost, 0)
    print(err_rate(L_pred, LTE), "%")


    ###################################################
    #############-----BINARY TASK - STANDARD----############
    ###################################################
    print("*"*100000, '\n')
    pi = np.ones((2, 1)) * 1.0/2.0
    print(pi)

    (m_0, c_0), (m_1, c_1), (m_2, c_2) = mean_cov_iris(DTR, LTR)

    D, L = load_iris()
    D = D[:, L != 0]
    L = L[L != 0]

    (DTR_bin, LTR_bin), (DTE_bin, LTE_bin) = split_db_2to1(D, L)

    print(DTE_bin.shape)
    print(DTR_bin.shape)

    (m_1, c_1), (m_2, c_2) = mean_cov_iris(DTR_bin, LTR_bin, bin=True)


    log_S = logpdf_GAU_ND(DTE_bin, m_1, c_1)
    log_S = np.vstack((log_S, logpdf_GAU_ND(DTE_bin, m_2, c_2)))
    llr = log_S[1, :] - log_S[0, :]


    threshold = -np.log(pi[1, :]/pi[0, :])

    predicted_labels = np.array([2 if l >= threshold else 1 for l in llr])

    print(err_rate(predicted_labels, LTE_bin), '%')

    ###################################################
    #############-----BINARY TASK - TIED----############
    ###################################################
    print("******" * 50)

    c = tied_cov(DTR_bin, LTR_bin, bin=True)

    log_S = logpdf_GAU_ND(DTE_bin, m_1, c)
    log_S = np.vstack((log_S, logpdf_GAU_ND(DTE_bin, m_2, c)))

    llr = log_S[1, :] - log_S[0, :]

    predicted_labels = np.array([2 if l >= threshold else 1 for l in llr])

    print(err_rate(predicted_labels, LTE_bin), "%")





