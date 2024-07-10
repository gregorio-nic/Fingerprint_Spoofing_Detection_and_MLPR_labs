import matplotlib.pyplot as plt
import numpy as np
import lab_05.multivariate_gaussian_model as mvg
import scipy


def confusion_matrix(L_pred, LTE, num_classes=2):
    M = np.zeros((num_classes, num_classes))
    for i, l in enumerate(L_pred):
        M[l][LTE[i]] += 1
    return M


def optimal_bayes_decisions_bin(scores, working_point, threshold=None):
    if threshold is None:
        threshold = -np.log(working_point[0]*working_point[1]/((1-working_point[0])*working_point[2]))
    pred_labels = [1 if s > threshold else 0 for s in scores]
    return pred_labels

def optimal_bayes_decisions_multiclass(loglikelihood, pi, cost_matrix, threshold=None):
    S = np.exp(loglikelihood)
    SJoint = S * pi

    SMarginal = mvg.vrow(SJoint.sum(0))

    SPost = SJoint / SMarginal

    C_cap = cost_matrix @ SPost

    L_pred = np.argmin(C_cap, axis=0)

    return L_pred

def false_neg_rate(confusion_matrix):
    return confusion_matrix[0][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])

def false_pos_rate(confusion_matrix):
    return confusion_matrix[1][0]/(confusion_matrix[1][0]+confusion_matrix[0][0])


def bayes_risk(working_point, confusion_matrix):
    FNR = false_neg_rate(confusion_matrix)
    FPR = false_pos_rate(confusion_matrix)
    return (working_point[0]*working_point[1]*FNR +
            (1-working_point[0])*working_point[2]*FPR)


def bayes_risk_norm(working_point, confusion_matrix):
    FNR = false_neg_rate(confusion_matrix)
    FPR = false_pos_rate(confusion_matrix)

    dummy = min(working_point[0]*working_point[1], (1-working_point[0])*working_point[2])
    DCFu = bayes_risk(working_point, confusion_matrix)

    return DCFu/dummy


def bayes_risk_multiclass(pi, confusion_matrix, cost_matrix):
    N = confusion_matrix.sum(axis=0)
    m = (confusion_matrix*cost_matrix).sum(axis=0)
    weight = pi / N.reshape((N.size, 1))
    DCF = m @ weight
    return DCF



def bayes_risk_norm_multiclass(pi, confusion_matrix, cost_matrix, num_classes=3):
    DCFu = bayes_risk_multiclass(pi, confusion_matrix, cost_matrix)
    dummy = np.min(cost_matrix@pi)

    return DCFu/dummy


def min_DCF(scores, LTE, working_point):
    thresholds = np.copy(scores)
    thresholds = np.append(thresholds, values=[np.inf, -np.inf])
    thresholds = np.sort(thresholds)

    min_dcf = np.inf
    for t in thresholds:
        L_pred = optimal_bayes_decisions_bin(scores, working_point, t)
        confM = confusion_matrix(L_pred, LTE)
        min_dcf = min(bayes_risk_norm(working_point, confM), min_dcf)

    return min_dcf


def plot_ROC(scores, LTE, working_point):
    thresholds = np.copy(scores)
    thresholds = np.append(thresholds, values=[np.inf, -np.inf])
    thresholds = np.sort(thresholds)

    tprs = np.array([])
    fprs = np.array([])
    plt.figure()

    for t in thresholds:
        L_pred = optimal_bayes_decisions_bin(scores, working_point, t)
        confM = confusion_matrix(L_pred, LTE)
        FNR = false_neg_rate(confM)
        FPR = false_pos_rate(confM)
        TPR = 1 - FNR
        fprs = np.append(fprs, FPR)
        tprs = np.append(tprs, TPR)

    plt.plot(fprs, tprs)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()


def bayes_error_plot(scores, LTE, comparing=False):
    if comparing is True:
        effPriorLogOdds = np.linspace(-4, 4, 21)
    else:
        effPriorLogOdds = np.linspace(-3, 3, 21)
    eff_priors = 1/(1+np.exp(-effPriorLogOdds))

    print(eff_priors)

    if comparing is True:

        plt.figure()

        for s in scores:
            DCF_norms = np.array([
                bayes_risk_norm(
                    working_point=(prior, 1, 1),
                    confusion_matrix=confusion_matrix(
                        L_pred=optimal_bayes_decisions_bin(scores[s], (prior, 1, 1)),
                        LTE=LTE
                    )
                ) for prior in eff_priors])

            min_DCFs = np.array([
                min_DCF(
                    scores=scores[s],
                    LTE=LTE,
                    working_point=(prior, 1, 1))
                for prior in eff_priors
            ])

            plt.plot(effPriorLogOdds, DCF_norms, label=f"DCF - {s}", color=np.random.rand(3,)%10, linewidth=2)
            plt.plot(effPriorLogOdds, min_DCFs, label=f"min DCF - {s}", color=np.random.rand(3,)%10, linewidth=2)
            plt.ylim([0, 1.1])
            plt.xlim([-3, 3])
            plt.xlabel('prior log-odds')
            plt.ylabel('DCF value')
            plt.legend(loc=3)

        plt.show()

    else:
        plt.figure()

        DCF_norms = np.array([
            bayes_risk_norm(
                working_point=(prior, 1, 1),
                confusion_matrix=confusion_matrix(
                    L_pred=optimal_bayes_decisions_bin(scores, (prior, 1, 1)),
                    LTE=LTE
                )
            ) for prior in eff_priors])

        min_DCFs = np.array([
            min_DCF(
                scores=scores,
                LTE=LTE,
                working_point=(prior, 1, 1))
            for prior in eff_priors
        ])

        plt.plot(effPriorLogOdds, DCF_norms, label=f"DCF", color='r', linewidth=2)
        plt.plot(effPriorLogOdds, min_DCFs, label=f"min DCF", color='b', linewidth=2)
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.xlabel('prior log-odds')
        plt.ylabel('DCF value')
        plt.legend(loc=3)

        plt.show()


if __name__ == '__main__':

    pi = np.ones((3, 1)) * 1 / 3
    D, L = mvg.load_iris()
    (DTR, LTR), (DTE, LTE) = mvg.split_db_2to1(D, L)
    print(f"DTR: {DTR.shape}, LTR: {LTR.shape}, DTE: {DTE.shape}, LTE: {LTE.shape}")

    ###################################################
    #################------STANDARD------##############
    ###################################################

    (m_0, c_0), (m_1, c_1), (m_2, c_2) = mvg.mean_cov_iris(DTR, LTR)

    log_S = mvg.logpdf_GAU_ND(DTE, m_0, c_0)
    log_S = np.vstack((log_S, mvg.logpdf_GAU_ND(DTE, m_1, c_1)))
    log_S = np.vstack((log_S, mvg.logpdf_GAU_ND(DTE, m_2, c_2)))

    log_SJoint = log_S + np.log(pi)

    log_SMarginal = mvg.vrow(scipy.special.logsumexp(log_SJoint, axis=0))

    log_SPost = log_SJoint - log_SMarginal

    SPost = np.exp(log_SPost)
    print(SPost.shape)

    L_pred = np.argmax(SPost, 0)
    print(mvg.err_rate(L_pred, LTE), "%")

    confM_std = confusion_matrix(L_pred, LTE, 3)

    print(confM_std)


    ###################################################
    ####################-----TIED----##################
    ###################################################

    c = mvg.tied_cov(DTR, LTR)

    S = np.exp(mvg.logpdf_GAU_ND(DTE, m_0, c))
    S = np.vstack((S, np.exp(mvg.logpdf_GAU_ND(DTE, m_1, c))))
    S = np.vstack((S, np.exp(mvg.logpdf_GAU_ND(DTE, m_2, c))))

    SJoint = S * pi

    SMarginal = mvg.vrow(SJoint.sum(0))

    SPost = SJoint / SMarginal

    L_pred = np.argmax(SPost, 0)
    print(mvg.err_rate(L_pred, LTE), "%")

    confM_tied = confusion_matrix(L_pred, LTE, 3)

    print(confM_tied)

    ###################################################
    ##################-----COMMEDIA----################
    ###################################################

    ll = np.load('Data/commedia_ll.npy')
    commedia_labels = np.load('Data/commedia_labels.npy')

    L_pred = np.argmax(ll, 0)
    confM_comm = confusion_matrix(L_pred, commedia_labels, 3)

    print(confM_comm)

    ###################################################
    ##############-----COMMEDIA-OPT DEC----############
    ###################################################
    pi = 0.8
    c_fn = 1
    c_fp = 10
    working_point = (pi, c_fn, c_fp)
    llr = np.load('Data/commedia_llr_infpar.npy')
    labels = np.load('Data/commedia_labels_infpar.npy')

    L_pred = optimal_bayes_decisions_bin(llr, working_point)

    confM = confusion_matrix(L_pred, labels)
    print(confM)

    ###################################################
    ###############-----BIN-TASK-EVAL----##############
    ###################################################

    DCFu = bayes_risk(working_point, confM)
    print(DCFu)

    DCF_norm = bayes_risk_norm(working_point, confM)

    print(DCF_norm)

    ###################################################
    ##################-----MIN_DCF----#################
    ###################################################

    DCF_min = min_DCF(scores=llr, LTE=labels, working_point=working_point)
    print(f"min_DCF: {DCF_min}")

    ###################################################
    ####################-----ROC----###################
    ###################################################

    plot_ROC(scores=llr, LTE=labels, working_point=working_point)

    ###################################################
    #############-----BAYES-ERROR-PLOTS----############
    ###################################################

    bayes_error_plot(llr, labels)

    ###################################################
    #############-----BAYES-ERROR-PLOTS----############
    ###################################################

    new_llr = np.load('Data/commedia_llr_infpar_eps1.npy')

    llrs = {
        'eps = 0.001': llr,
        'eps = 1': new_llr
    }

    bayes_error_plot(llrs, labels, comparing=True)

    ###################################################
    ##############-----MULTICLASS-EVAL----#############
    ###################################################

    ll = np.load('Data/commedia_ll.npy')
    labels = np.load('Data/commedia_labels.npy')

    pi = np.array([[0.3], [0.4], [0.3]])
    C = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])


    L_pred = optimal_bayes_decisions_multiclass(ll, pi, C)

    print(L_pred)

    confM = confusion_matrix(L_pred, labels, 3)

    print(confM)


    DCF = bayes_risk_multiclass(pi, confM, C)

    DCF_norm = bayes_risk_norm_multiclass(pi, confM, C, 3)

    print(f"DCF: {DCF}\nDCF_norm: {DCF_norm}")














