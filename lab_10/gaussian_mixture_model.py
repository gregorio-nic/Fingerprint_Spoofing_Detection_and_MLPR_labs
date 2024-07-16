import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import lab_04.gaussian_density_estimation as gde
import GMM_load
import lab_07.measuring_predictions as mp


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


# computes the log density of a GMM for a set of samples contained in matrix X
def logpdf_GMM(X, gmm):
    log_SJoint = []
    for i, g in enumerate(gmm):
        log_SJoint.append(gde.logpdf_GAU_ND(X, g[1], g[2]) + np.log(g[0]))

    log_SJoint = np.vstack(log_SJoint)
    return log_SJoint

def log_marginal(log_SJoint):
    return scipy.special.logsumexp(log_SJoint, axis=0)

def responsibilities(log_SJoint, log_SMarginal):
    return np.exp(log_SJoint-log_SMarginal)

def compute_Z(gamma):
    return gamma.sum()

def compute_F(X, gamma):
    return gde.vcol((gamma * X).sum(1))

def compute_S(X, gamma):
    return (gde.vrow(gamma) * X) @ X.T

def new_mus(F, Z):
    return F / Z

def new_covs(S, Z, mu):
    return (S / Z) - (mu @ mu.T)

def new_ws(X, Z):
    return Z / X.shape[1]

def E_step(log_SJoint, log_SMarginal):
    return responsibilities(log_SJoint, log_SMarginal)


def smooth_covariance_matrix(cov, psi):
    U, s, _ = np.linalg.svd(cov)
    s[s < psi] = psi
    cov = U @ (gde.vcol(s) * U.T)
    return cov


def M_step(X, gammas, psi=None, covType='full'):
    new_gmm = []
    for gamma in gammas:

        Z = compute_Z(gamma)
        F = compute_F(X, gamma)
        S = compute_S(X, gamma)

        mu = new_mus(F, Z)
        w = new_ws(X, Z)
        cov = new_covs(S, Z, mu)
        if covType== 'diagonal':
            cov = cov * np.eye(X.shape[0])

        new_gmm.append((w, mu, cov))

    if covType == 'tied':
        CTied = 0
        for w, mu, C in new_gmm:
            CTied += w * C
        new_gmm = [(w, mu, CTied) for w, mu, C in new_gmm]

    if psi is not None:
        new_gmm = [(w, mu, smooth_covariance_matrix(cov, psi)) for w, mu, cov in new_gmm]
    return new_gmm


def LBG(gmm, alpha=0.1):
    gmmOut = []
    for (w, mu, C) in gmm:
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut

def EM_algorithm(D, gmm, delta_l=1e-6, psi=None, covType='full'):
    log_SJoint = logpdf_GMM(D, gmm)
    log_SMarginal = log_marginal(log_SJoint)

    prec_ll = -np.inf

    while ((log_SMarginal.mean()) - prec_ll) > delta_l:
        prec_ll = log_SMarginal.mean()

        gammas = E_step(log_SJoint, log_SMarginal)
        new_gmm = M_step(D, gammas, psi, covType)

        log_SJoint = logpdf_GMM(D, new_gmm)
        log_SMarginal = log_marginal(log_SJoint)

    log_SJoint = logpdf_GMM(D, new_gmm)
    log_SMarginal = log_marginal(log_SJoint)
    #print(log_SMarginal.mean())
    return new_gmm

def LBG_EM(X, numComponents, delta_l=1e-6, alpha=0.1, psi=None, covType='full'):
    mu = gde.vcol(X.mean(1))
    C = np.array(((X - mu) @ (X - mu).T) / float(X.shape[1]))

    if covType == 'diagonal':
        C = C * np.eye(X.shape[0])

    if psi is not None:
        C = smooth_covariance_matrix(C, psi)
    gmm = [(1.0, mu, C)]

    while len(gmm) < numComponents:
        # Split the components
        gmm = LBG(gmm, alpha)
        # Run the EM for the new GMM
        gmm = EM_algorithm(X, gmm, delta_l, psi, covType)
    return gmm

def GMM(DTR, LTR, DTE, numComponents, covType, psi=0.01):
    gmm0 = LBG_EM(DTR[:, LTR == 0], numComponents=numComponents, covType=covType, psi=psi)
    gmm1 = LBG_EM(DTR[:, LTR == 1], numComponents=numComponents, covType=covType, psi=psi)

    scores = log_marginal(logpdf_GMM(DTE, gmm1)) - log_marginal(logpdf_GMM(DTE, gmm0))

    return scores

if __name__ == '__main__':
    '''
    # 4D check
    D = np.load('Data/GMM_data_4D.npy')
    gmm = GMM_load.load_gmm('Data/GMM_4D_3G_init.json')

    log_SJoint = logpdf_GMM(D, gmm)
    log_SMarginal = log_marginal(log_SJoint)

    check = np.load('Data/GMM_4D_3G_init_ll.npy')

    print(log_SMarginal - check)

    # 1D check
    D = np.load('Data/GMM_data_1D.npy')
    gmm = GMM_load.load_gmm('Data/GMM_1D_3G_init.json')

    log_SJoint = logpdf_GMM(D, gmm)
    log_SMarginal = log_marginal(log_SJoint)

    check = np.load('Data/GMM_1D_3G_init_ll.npy')

    print(log_SMarginal - check)
    '''

    ################################################
    ################# EM ALGO-4D ####################
    ################################################
    '''

    D = np.load('Data/GMM_data_4D.npy')
    gmm = GMM_load.load_gmm('Data/GMM_4D_3G_init.json')



    final_gmm = EM_algorithm(D, gmm)

    ################################################
    ################# EM ALGO-1D ####################
    ################################################

    D = np.load('Data/GMM_data_1D.npy')
    gmm = GMM_load.load_gmm('Data/GMM_1D_3G_init.json')

    final_gmm = EM_algorithm(D, gmm)
    print(final_gmm)

    plt.figure()
    plt.hist(D.ravel(), bins=25, density=True)
    XPlot_1D = np.linspace(-10, 5, 1000)
    log_SJoint = logpdf_GMM(gde.vrow(XPlot_1D), final_gmm)
    log_SMarginal = log_marginal(log_SJoint)
    plt.plot(XPlot_1D.ravel(), np.exp(log_SMarginal.ravel()))
    plt.show()
    

    ################################################
    #################### LBG-4D ####################
    ################################################

    D = np.load('Data/GMM_data_4D.npy')
    final_gmm = LBG_EM(D, 4)

    ################################################
    #################### LBG-1D ####################
    ################################################

    D = np.load('Data/GMM_data_1D.npy')
    gmm = GMM_load.load_gmm('Data/GMM_1D_3G_init.json')

    final_gmm = LBG_EM(D, 4)
    print(final_gmm)

    plt.figure()
    plt.hist(D.ravel(), bins=25, density=True)
    XPlot_1D = np.linspace(-10, 5, 1000)
    log_SJoint = logpdf_GMM(gde.vrow(XPlot_1D), final_gmm)
    log_SMarginal = log_marginal(log_SJoint)
    plt.plot(XPlot_1D.ravel(), np.exp(log_SMarginal.ravel()))
    plt.show()


    ################################################
    #################### LBG-4D ####################
    ################################################
    D = np.load('Data/GMM_data_1D.npy')
    gmm = GMM_load.load_gmm('Data/GMM_1D_3G_init.json')
    psi = 0.01
    final_gmm = LBG_EM(D, 4, psi=psi)
    print(final_gmm)
    '''
    ################################################
    ##################### IRIS #####################
    ################################################
    (D, L) = load_iris()['data'].T, load_iris()['target']  # D è le matrice di dati -> ciascuna colonna è un set di dati
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)  ### Split DB into training & evaluation

    for Type in ['full', 'diagonal', 'tied']:
        for n in [1, 2, 4, 8, 16]:
            gmm0 = LBG_EM(DTR[:, LTR == 0], numComponents = n, covType=Type, psi=0.01)
            gmm1 = LBG_EM(DTR[:, LTR == 1], numComponents = n, covType=Type, psi=0.01)
            gmm2 = LBG_EM(DTR[:, LTR == 2], numComponents = n, covType=Type, psi=0.01)

            prior_probability = gde.vcol(np.ones(3) / 3.0)
            scores = [log_marginal(logpdf_GMM(DTE, gmm0)), log_marginal(logpdf_GMM(DTE, gmm1)),
                        log_marginal(logpdf_GMM(DTE, gmm2))]  # Class-conditional log-likelihoods
            scores = np.vstack(scores) + np.log(prior_probability)  # We add the log-prior to get the log-joint

            L_pred = scores.argmax(0)  # Predictions
            print(
                f"Covariance Matrix: {Type}, number Gaussians: {n} - Error rate: {mp.err_rate(L_pred, LTE)}%")