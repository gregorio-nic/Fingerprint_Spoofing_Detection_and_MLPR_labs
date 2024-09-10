import numpy as np
import scipy
import matplotlib.pyplot as plt


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


def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    mu = vcol(D.mean(1))
    for i in np.unique(L):
        D_class = D[:, L == i]
        mu_class = vcol(D_class.mean(1))
        Sb += (mu_class - mu) @ (mu_class - mu).T * D_class.shape[1]
        Sw += (D_class - mu_class) @ (D_class - mu_class).T
    return Sb / D.shape[1], Sw / D.shape[1]


def compute_lda(D, L, m):
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]


def apply_lda(U, D):
    return U.T @ D

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