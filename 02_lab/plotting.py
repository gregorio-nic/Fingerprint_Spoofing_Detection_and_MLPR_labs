import numpy as np
import matplotlib.pyplot as plt

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

def plot_hist(D, L):
    Feature = {}

    for feature in range(D.shape[0]):
        Feature[feature] = "Feature {}".format(feature + 1)

    L_0 = (L == 0)
    L_1 = (L == 1)
    L_2 = (L == 2)

    # creating a separate figure for each feature
    for f in Feature:

        plt.figure()

        plt.xlabel(Feature[f])

        plt.hist(D[f, L_0], bins=10, density=True, alpha=0.4, label='Setosa')
        plt.hist(D[f, L_1], bins=10, density=True, alpha=0.4, label='Versicolor')
        plt.hist(D[f, L_2], bins=10, density=True, alpha=0.4, label='Virginica')
        plt.legend()
        plt.show()


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


def project_plot_hist(D, L):
    Feature = {}

    for feature in range(D.shape[0]):
        Feature[feature] = "Feature {}".format(feature + 1)

    Data_Spoofed = (L == 0)
    Data_Authentic = (L == 1)

    for f in Feature:

        plt.figure()  # creo una figura dedicata per ciascuna proprietà

        plt.xlabel(Feature[f])  # nome della proprità analizzata su asse x

        plt.hist(D[f, Data_Spoofed], bins=100, density=True, alpha=0.4, rwidth=0.9, label='Spoofed Fingerprint')
        plt.hist(D[f, Data_Authentic], bins=100, density=True, alpha=0.4, rwidth=0.9, label='Authentic Fingerprint')
        plt.legend()
        # plt.tight_layout() -> a cosa serve??
        plt.show()

def project_plot_scatter(D, L):
    Feature = {}

    for feature in range(D.shape[0]):
        Feature[feature] = "Feature {}".format(feature+1)

    Data_Spoofed = (L == 0)
    Data_Authentic = (L == 1)

    for feature_x in Feature:
        for feature_y in Feature:
            if feature_y == feature_x:  # La figura presenta la relazione tra proprietà.
                continue  # Solo il caso in cui (x, y) con asse_x=asse_y, non è rilevante, ma non deve essere bloccante
            plt.figure()
            plt.xlabel(Feature[feature_x])    # nome della proprietà analizzata su asse x
            plt.ylabel(Feature[feature_y])    # nome della proprietà analizzata su asse x
            plt.scatter(D[feature_x, Data_Spoofed], D[feature_y, Data_Spoofed], label='Spoofed', alpha=0.2)
            plt.scatter(D[feature_x, Data_Authentic], D[feature_y, Data_Authentic], label='Authentic', alpha=0.2)
            plt.legend()

            plt.show()



if __name__ == '__main__':

    D, L = load('../iris.csv')
    print('D: ', D)
    print('L: ', L)

    #plot_hist(D, L)
    plot_scatter(D, L)