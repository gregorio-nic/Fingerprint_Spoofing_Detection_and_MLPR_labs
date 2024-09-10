import numpy as np

import matplotlib.pyplot as plt
import lab_03.dimensionality_reduction as dr
import lab_02.plotting as p
import lab_05.multivariate_gaussian_model as mvg

if __name__ == '__main__':
    data = np.loadtxt('../Project/trainData.txt', delimiter=",").T

    D, L = data[:data.shape[0] - 1, :], data[data.shape[0] - 1, :]

    '''
    mu = dr.vcol(np.mean(D, axis=1))
    D = dr.center_data(D, mu)
    C = 1 / D.shape[1] * D @ D.T

    num_dim = 6
    P = dr.pca(C, num_dim)
    D_reduced = P.T @ D
    p.project_plot_hist(D_reduced, L)
    
    U = dr.compute_lda(D, L, m=1)

    D_reduced = dr.apply_lda(U, D)
    p.project_plot_hist(D_reduced, L)
    '''
    #### CLASSIFICATION

    (DTR, LTR), (DTE, LTE) = dr.split_db_2to1(D, L)

    U = dr.compute_lda(DTR, LTR, m=1)

    DTR_projected = dr.apply_lda(U, DTR)
    DTE_projected = dr.apply_lda(U, DTE)

    threshold = (DTR_projected[0, LTR==0].mean() + DTR_projected[0, LTR==1].mean()) / 2.0

    PVAL = np.zeros(shape=LTE.shape, dtype=np.int32)
    PVAL[DTE_projected[0] >= threshold] = 1
    PVAL[DTE_projected[0] < threshold] = 0

    error_rate = mvg.err_rate(PVAL, LTE)
    print(error_rate)

    mu = dr.vcol(np.mean(DTR, axis=1))

    D = dr.center_data(D, mu)

    (DTRC, LTR), (DTEC, LTE) = dr.split_db_2to1(D, L)

    C = 1 / DTRC.shape[1] * DTRC @ DTRC.T

    pca_dims = [1, 2, 3, 4, 5, 6]
    for pca in pca_dims:
        P = dr.pca(C, dim=pca)

        DTRP = P.T @ DTRC
        DTEP = P.T @ DTEC

        if pca == 2:
            U = - dr.compute_lda(DTRP, LTR, m=1)
        else:
            U = dr.compute_lda(DTRP, LTR, m=1)

        DTR_projected = dr.apply_lda(U, DTRP)
        DTE_projected = dr.apply_lda(U, DTEP)

        threshold = (DTR_projected[0, LTR == 0].mean() + DTR_projected[0, LTR == 1].mean()) / 2.0

        PVAL = np.zeros(shape=LTE.shape, dtype=np.int32)
        PVAL[DTE_projected[0] >= threshold] = 1
        PVAL[DTE_projected[0] < threshold] = 0

        error_rate = mvg.err_rate(PVAL, LTE)
        print(f"pca: {pca} error rate:{error_rate}%")
