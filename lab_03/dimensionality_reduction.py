import numpy as np

import matplotlib.pyplot as plt
import sys


'''turns vector into row-matrix'''
def vrow(vector):
    return vector.reshape(1, vector.size)


'''turns vector into col-matrix'''
def vcol(vector):
    return vector.reshape(vector.size, 1)

def load(file_name):
    with open(file_name) as f:
        d = {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2
        }
        first = 0
        for line in f:
            a = np.array(line.split(",")[0:4], dtype=np.float32)
            b = np.array(d[line.split(",")[4].rstrip()])

            a = a.reshape(4, 1)
            if first == 1:
                D = np.hstack([D, a])
                L = np.hstack([L, b])
            else:
                D = a
                L = b
                first = 1
    return D, L


def plot_scatter(D, L):
    Feature = {}

    for feature in range(D.shape[0]):
        Feature[feature] = "Feature {}".format(feature+1)

    L_0 = (L == 0)
    L_1 = (L == 1)
    L_2 = (L == 2)

    for feature_x in Feature:
        for feature_y in Feature:
            if feature_y == feature_x:  # La figura presenta la relazione tra proprietà.
                continue  # Solo il caso in cui (x, y) con asse_x=asse_y, non è rilevante, ma non deve essere bloccante
            plt.figure()
            plt.xlabel(Feature[feature_x])    # nome della proprietà analizzata su asse x
            plt.ylabel(Feature[feature_y])    # nome della proprietà analizzata su asse x
            plt.scatter(D[feature_x, L_0], D[feature_y, L_0], label='Setosa', alpha=0.8)
            plt.scatter(D[feature_x, L_1], D[feature_y, L_1], label='Versicolor', alpha=0.8)
            plt.scatter(D[feature_x, L_2], D[feature_y, L_2], label='Virginica', alpha=0.8)
            plt.legend()

            plt.show()


def pca(C, dim):
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:dim]
    return P


def center_data(D, mu):
    return D - mu


def lda(D, L):
    L_0 = L == 0
    L_1 = L == 1
    L_2 = L == 2

    print(D[:, L_0].shape, D.shape)

    for l in range(3):
        D_filtered = D[:, L==l]
        C = 1/D_filtered.shape[1]


if __name__ == '__main__':
    D, L = load('../iris.csv')

    mu = vcol(np.mean(D, axis=1))

    DC = center_data(D, mu)

    C = 1/D.shape[1] * DC @ DC.T


    # PCA
    P = pca(C, dim=2)

    DP = P.T @ DC

    solution = np.load('Solution/IRIS_PCA_matrix_m4.npy')

    plot_scatter(DP, L)

    print(f"P: {P}")


    #LDA
    lda(D, L)