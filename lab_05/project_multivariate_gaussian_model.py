import numpy as np
import lab_05.multivariate_gaussian_model as mvg
import matplotlib.pyplot as plt
import lab_03.dimensionality_reduction as dr

np.set_printoptions(linewidth=np.inf)


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

    return (DTR, LTR), (DTE, LTE.astype(int))


def mean(DTR, LTR):
    L_0 = (LTR == 0)
    L_1 = (LTR == 1)

    m_0 = mvg.vcol(np.mean(DTR[:, L_0], axis=1))
    m_1 = mvg.vcol(np.mean(DTR[:, L_1], axis=1))

    return m_0, m_1

def cov(DTR, LTR, m_0, m_1):
    L_0 = (LTR == 0)
    L_1 = (LTR == 1)

    c_0 = 1 / (DTR[:, L_0].shape[1]) * (DTR[:, L_0] - m_0) @ (DTR[:, L_0] - m_0).T
    c_1 = 1 / (DTR[:, L_1].shape[1]) * (DTR[:, L_1] - m_1) @ (DTR[:, L_1] - m_1).T

    return c_0, c_1

def tied_cov(DTR, LTR, m_0, m_1):
    L_0 = (LTR == 0)
    L_1 = (LTR == 1)

    c_0 = (DTR[:, L_0] - m_0) @ (DTR[:, L_0] - m_0).T
    c_1 = (DTR[:, L_1] - m_1) @ (DTR[:, L_1] - m_1).T

    c = 1/(DTR.shape[1]) * (c_0 + c_1)
    return c

def corr(C):
    return C/(mvg.vcol(C.diagonal()**0.5) * mvg.vrow(C.diagonal()**0.5))

def plot_heatmap(C, title, color="Reds"):
    plt.figure(title)
    plt.imshow(C, cmap=color, interpolation='nearest')
    plt.xticks(np.arange(C.shape[1]))
    plt.yticks(np.arange(C.shape[1]))
    plt.colorbar()  # Aggiunge una barra dei colori per la scala
    plt.title(title)
    plt.show(block=False)

def MVG_standard(DTR, LTR, DTE):

    m_0, m_1 = mean(DTR, LTR)
    c_0, c_1 = cov(DTR, LTR, m_0, m_1)

    corr_0 = corr(c_0)
    corr_1 = corr(c_1)


    print("\n")
    print("*"*100)
    print("STANDARD")
    print("*" * 100)
    print(f"Covariance matrices\nCov_0 =\n{c_0}\nCov_1 =\n{c_1}\n")
    print(f"Correlation matrices\nCorr_0 =\n{corr_0}\nCorr_1 =\n{corr_1}")

    plot_heatmap(corr_0, "Standard - C_0")
    plot_heatmap(corr_1, "Standard - C_1")

    log_S = mvg.logpdf_GAU_ND(DTE, m_0, c_0)
    log_S = np.vstack((log_S, mvg.logpdf_GAU_ND(DTE, m_1, c_1)))
    llr = log_S[1, :] - log_S[0, :]

    return llr
def MVG_tied(DTR, LTR, DTE):

    m_0, m_1 = mean(DTR, LTR)
    c = tied_cov(DTR, LTR, m_0, m_1)

    corr_0 = corr(c)
    plot_heatmap(corr_0, "Tied - C")

    print("\n")
    print("*" * 100)
    print("TIED")
    print("*" * 100)
    print(f"Covariance matrix\nCov =\n{c}\n")
    print(f"Correlation matrix\nCorr =\n{corr(c)}")
    log_S = mvg.logpdf_GAU_ND(DTE, m_0, c)
    log_S = np.vstack((log_S, mvg.logpdf_GAU_ND(DTE, m_1, c)))

    llr = log_S[1, :] - log_S[0, :]

    return llr


def MVG_naive(DTR, LTR, DTE):

    m_0, m_1 = mean(DTR, LTR)
    c_0, c_1 = cov(DTR, LTR, m_0, m_1)

    identity = np.eye(DTR.shape[0])

    nb_c0 = c_0 * identity
    nb_c1 = c_1 * identity

    nb_corr_0 = corr(nb_c0)
    nb_corr_1 = corr(nb_c1)
    plot_heatmap(nb_corr_0, "NaiveBayes - C_0")
    plot_heatmap(nb_corr_1, "NaiveBayes - C_1")


    print("\n")
    print("*" * 100)
    print("NAIVE-BAYES")
    print("*" * 100)
    print(f"Covariance matrices\nCov_0 =\n{nb_c0}\nCov_1 =\n{nb_c1}\n")
    print(f"Correlation matrices\nCorr_0 =\n{corr(nb_c0)}\nCorr_1 =\n{corr(nb_c1)}")
    log_S = mvg.logpdf_GAU_ND(DTE, m_0, nb_c0)
    log_S = np.vstack((log_S, mvg.logpdf_GAU_ND(DTE, m_1, nb_c1)))

    llr = log_S[1, :] - log_S[0, :]

    return llr


if __name__ == '__main__':
    data = np.loadtxt('../Project/trainData.txt', delimiter=",").T

    D, L = data[:data.shape[0] - 1, :], data[data.shape[0] - 1, :]

    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    llr = MVG_standard(DTR, LTR, DTE)

    pi = np.ones((2, 1)) * (1.0/2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Standard error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

    llr = MVG_tied(DTR, LTR, DTE)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Tied error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

    llr = MVG_naive(DTR, LTR, DTE)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Naive error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

    ### Discarding features 4 and 5
    '''DTR, DTE = DTR[:4:, :], DTE[:4:, :]


    llr = MVG_standard(DTR, LTR, DTE)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Standard error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

    llr = MVG_tied(DTR, LTR, DTE)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Tied error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

    llr = MVG_naive(DTR, LTR, DTE)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Naive error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

    ### USING only features 1 and 2
    
    DTR, DTE = DTR[:2:, :], DTE[:2:, :]

    llr = MVG_standard(DTR, LTR, DTE)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Standard error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

    llr = MVG_tied(DTR, LTR, DTE)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Tied error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

    llr = MVG_naive(DTR, LTR, DTE)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Naive error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")
    
    ### USING only features 3 and 4
    DTR, DTE = DTR[2:4:, :], DTE[2:4:, :]

    llr = MVG_standard(DTR, LTR, DTE)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Standard error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

    llr = MVG_tied(DTR, LTR, DTE)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Tied error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

    llr = MVG_naive(DTR, LTR, DTE)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Naive error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")
    '''

    ### PCA

    mu = dr.vcol(np.mean(DTR, axis=1))

    D = dr.center_data(D, mu)

    (DTRC, LTR), (DTEC, LTE) = split_db_2to1(D, L)

    C = 1 / DTRC.shape[1] * DTRC @ DTRC.T

    # PCA
    P = dr.pca(C, dim=6)

    DTRP = P.T @ DTRC
    DTEP = P.T @ DTEC

    llr = MVG_standard(DTRP, LTR, DTEP)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Standard error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

    llr = MVG_tied(DTRP, LTR, DTEP)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Tied error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

    llr = MVG_naive(DTRP, LTR, DTEP)

    pi = np.ones((2, 1)) * (1.0 / 2.0)

    threshold = -np.log(pi[1, :] / pi[0, :])

    predicted_labels = np.array([1 if l >= threshold else 0 for l in llr])

    print(f"Naive error rate = {mvg.err_rate(predicted_labels, LTE)} %")
    print("\n")

