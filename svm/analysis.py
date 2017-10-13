import argparse
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn import svm
from numpy import where

class Numbers:
    """
    Class to store MNIST data
    """
    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.
        import pickle, gzip
        # Load the dataset
        f = gzip.open(location, 'rb')

        train_set, valid_set, test_set = pickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

        f.close()
        # Filter 3's and 8's from train and testing data
        idx_train = where((self.train_y == 3)|(self.train_y == 8))[0]
        self.train_x, self.train_y = self.train_x[idx_train], self.train_y[idx_train]

        idx_test = where((self.test_y == 3)|(self.test_y == 8))[0]
        self.test_x, self.test_y = self.test_x[idx_test], self.test_y[idx_test]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='svm classifier arguments')
    parser.add_argument('--c', type=float, default=1.0,
                        help="Penalty parameter C of the error term")
    parser.add_argument('--kernel', type=str, default="linear",
                        help="Specifies the kernel type to be used in the algorithm")
    parser.add_argument('--degree', type=int, default=3,
                        help="Degree of the polynomial kernel function (‘poly’)")
    parser.add_argument('--plot', type=bool, default=True,
                        help="Plot few Support vectors")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    classifier = svm.SVC(C=args.c, kernel=args.kernel, degree=args.degree)

    classifier.fit(data.train_x, data.train_y)

    accuracy = classifier.score(data.test_x,data.test_y)

    print("#sAccuracy: {:2.4f} #Kernel: {} #Parameter_C: {:1.0f}".format(100*accuracy,args.kernel,args.c))

    if args.plot:
        for i in range(1,11):
            plt.subplot(2,5, i)
            if i < 6:
                plt.title('Support Vector:{}'.format(3),fontsize=8)
                plt.imshow(classifier.support_vectors_[i-1].reshape(28, 28), cmap=cm.gray)
            else:
                plt.title('Support Vector:{}'.format(8),fontsize=8)
                plt.imshow(classifier.support_vectors_[5-i].reshape(28, 28), cmap=cm.gray)
        plt.show()
