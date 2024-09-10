import numpy as np
import measuring_predictions as mp
import lab_05.project_multivariate_gaussian_model as mvg
import lab_03.dimensionality_reduction as dr


if __name__ == '__main__':
    data = np.loadtxt('../Project/trainData.txt', delimiter=",").T

    D, L = data[:data.shape[0] - 1, :], data[data.shape[0] - 1, :]

    (DTR, LTR), (DTE, LTE) = mvg.split_db_2to1(D, L)

    prior = 0.5
    C_fn = 1.0
    C_fp = 1.0

    working_points = [(0.5, 1.0, 1.0),
                      (0.9, 1.0, 1.0),
                      (0.1, 1.0, 1.0),
                      (0.5, 1.0, 9.0),
                      (0.5, 9.0, 1.0)
                      ]

    pca_dims = [1, 2, 3, 4, 5, 6]


    for wp in working_points:
        eff_prior = (wp[0] * wp[1]) / ((wp[0] * wp[1]) + ((1-wp[0]) * wp[2]))
        print(f"Working_point: {wp} - Effective prior: {eff_prior}")

    new_working_points = working_points[:3]



    DTR_mean = dr.vcol(np.mean(DTR, axis = 1))
    DTE_mean = dr.vcol(np.mean(DTE, axis=1))
    DTR_centered = dr.center_data(DTR, DTR_mean)
    DTE_centered = dr.center_data(DTE, DTE_mean)
    DTR_cov = 1/DTR_centered.shape[1] * DTR_centered @ DTR_centered.T

    llr_std = mvg.MVG_standard(DTR, LTR, DTE)
    llr_naive = mvg.MVG_naive(DTR, LTR, DTE)
    llr_tied = mvg.MVG_tied(DTR, LTR, DTE)
    '''
    for wp in working_points:

        eff_prior = (wp[0] * wp[1]) / ((wp[0] * wp[1]) + ((1 - wp[0]) * wp[2]))

        L_pred_std = mp.optimal_bayes_decisions_bin(llr_std, wp)
        confM_std = mp.confusion_matrix(L_pred_std, LTE)
        DCF_norm_std = mp.bayes_risk_norm(wp, confM_std)
        minDCF_std = mp.min_DCF(llr_std, LTE, wp)

        L_pred_naive = mp.optimal_bayes_decisions_bin(llr_naive, wp)
        confM_naive = mp.confusion_matrix(L_pred_naive, LTE)
        DCF_norm_naive = mp.bayes_risk_norm(wp, confM_naive)
        minDCF_naive = mp.min_DCF(llr_naive, LTE, wp)

        L_pred_tied = mp.optimal_bayes_decisions_bin(llr_tied, wp)
        confM_tied = mp.confusion_matrix(L_pred_tied, LTE)
        DCF_norm_tied = mp.bayes_risk_norm(wp, confM_tied)
        minDCF_tied = mp.min_DCF(llr_tied, LTE, wp)
        print(f"--------------- Working_point: {wp} - Effective prior: {eff_prior}  ---------------\n"
              f"Confusion matrix standard version:\n{confM_std}\n"
              f"DCF_norm std: {DCF_norm_std}, minDCF_std: {minDCF_std}\n"
              f"Confusion matrix naive version:\n{confM_naive}\n"
              f"DCF_norm naive: {DCF_norm_naive}, minDCF_naive: {minDCF_naive}\n"
              f"Confusion matrix tied version:\n{confM_tied}\n"
              f"DCF_norm tied: {DCF_norm_tied}, minDCF_tied: {minDCF_tied}"
              )

    for num_dim in pca_dims:
        P = dr.pca(DTR_cov, num_dim)
        DTR_reduced = P.T @ DTR_centered
        DTE_reduced = P.T @ DTE_centered

        llr_std = mvg.MVG_standard(DTR_reduced, LTR, DTE_reduced)
        llr_naive = mvg.MVG_naive(DTR_reduced, LTR, DTE_reduced)
        llr_tied = mvg.MVG_tied(DTR_reduced, LTR, DTE_reduced)


        for wp in new_working_points:
            eff_prior = (wp[0] * wp[1]) / ((wp[0] * wp[1]) + ((1 - wp[0]) * wp[2]))

            L_pred_std = mp.optimal_bayes_decisions_bin(llr_std, wp)
            confM_std = mp.confusion_matrix(L_pred_std, LTE)
            DCF_norm_std = mp.bayes_risk_norm(wp, confM_std)
            minDCF_std = mp.min_DCF(llr_std, LTE, wp)

            L_pred_naive = mp.optimal_bayes_decisions_bin(llr_naive, wp)
            confM_naive = mp.confusion_matrix(L_pred_naive, LTE)
            DCF_norm_naive = mp.bayes_risk_norm(wp, confM_naive)
            minDCF_naive = mp.min_DCF(llr_naive, LTE, wp)

            L_pred_tied = mp.optimal_bayes_decisions_bin(llr_tied, wp)
            confM_tied = mp.confusion_matrix(L_pred_tied, LTE)
            DCF_norm_tied = mp.bayes_risk_norm(wp, confM_tied)
            minDCF_tied = mp.min_DCF(llr_tied, LTE, wp)
            print(f"--------------- Working_point: {wp} - Effective prior: {eff_prior} - PCA: {num_dim} ---------------\n"
                  f"Confusion matrix standard version:\n{confM_std}\n"
                  f"DCF_norm std: {DCF_norm_std}, minDCF_std: {minDCF_std}\n"
                  f"Confusion matrix naive version:\n{confM_naive}\n"
                  f"DCF_norm naive: {DCF_norm_naive}, minDCF_naive: {minDCF_naive}\n"
                  f"Confusion matrix tied version:\n{confM_tied}\n"
                  f"DCF_norm tied: {DCF_norm_tied}, minDCF_tied: {minDCF_tied}"
                  )


    '''
    ######################################################################################
    ##########################pi_tilde=0.1 - BEST_PCA=3 ##########################
    ######################################################################################

    wp = working_points[2]
    num_dim = 3

    P = dr.pca(DTR_cov, num_dim)
    DTR_reduced = P.T @ DTR_centered
    DTE_reduced = P.T @ DTE_centered

    llr_std = mvg.MVG_standard(DTR_reduced, LTR, DTE_reduced)
    llr_naive = mvg.MVG_naive(DTR_reduced, LTR, DTE_reduced)
    llr_tied = mvg.MVG_tied(DTR_reduced, LTR, DTE_reduced)

    eff_prior = (wp[0] * wp[1]) / ((wp[0] * wp[1]) + ((1 - wp[0]) * wp[2]))

    L_pred_std = mp.optimal_bayes_decisions_bin(llr_std, wp)
    confM_std = mp.confusion_matrix(L_pred_std, LTE)
    DCF_norm_std = mp.bayes_risk_norm(wp, confM_std)
    minDCF_std = mp.min_DCF(llr_std, LTE, wp)

    L_pred_naive = mp.optimal_bayes_decisions_bin(llr_naive, wp)
    confM_naive = mp.confusion_matrix(L_pred_naive, LTE)
    DCF_norm_naive = mp.bayes_risk_norm(wp, confM_naive)
    minDCF_naive = mp.min_DCF(llr_naive, LTE, wp)

    L_pred_tied = mp.optimal_bayes_decisions_bin(llr_tied, wp)
    confM_tied = mp.confusion_matrix(L_pred_tied, LTE)
    DCF_norm_tied = mp.bayes_risk_norm(wp, confM_tied)
    minDCF_tied = mp.min_DCF(llr_tied, LTE, wp)
    print(f"--------------- Working_point: {wp} - Effective prior: {eff_prior} - PCA: {num_dim} ---------------\n"
          f"Confusion matrix standard version:\n{confM_std}\n"
          f"DCF_norm std: {DCF_norm_std}, minDCF_std: {minDCF_std}\n"
          f"Confusion matrix naive version:\n{confM_naive}\n"
          f"DCF_norm naive: {DCF_norm_naive}, minDCF_naive: {minDCF_naive}\n"
          f"Confusion matrix tied version:\n{confM_tied}\n"
          f"DCF_norm tied: {DCF_norm_tied}, minDCF_tied: {minDCF_tied}"
          )


    scores = {
        'standard': llr_std,
        'naive': llr_naive,
        'tied': llr_tied
    }

    mp.bayes_error_plot(scores, LTE, comparing=True)
