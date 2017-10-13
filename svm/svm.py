__author__ = "Ravi Bhadeshiya"
__email__ = "ravib@terpmail.umd.edu"

import argparse
from numpy import array, sum, where
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn import svm

kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector.
    """
    # w = zeros(len(x[0]))
    # DONE: IMPLEMENT THIS FUNCTION
    return sum(x*y.reshape((y.size, 1))*alpha.reshape((alpha.size, 1)),axis=0)


def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a primal support vector, return the indices for all of the support
    vectors
    """
    # support = set()
    # DONE: IMPLEMENT THIS FUNCTION
    return set([iter for iter in range(len(x))
                if abs(((sum(w*x[iter])+b)*y[iter])-1) < tolerance])


def find_slack(x, y, w, b):
    """
    Given a primal support vector instance, return the indices for all of the
    slack vectors
    """
    # slack = set()
    # DONE: IMPLEMENT THIS FUNCTION
    return set([iter for iter in range(len(x))
                if (sum(w*x[iter])+b)*y[iter] < 0])

########################################################## Analysis Part ##
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

        # Filter 3's and 8's from train data
        idx_train = where((self.train_y == 3)|(self.train_y == 8))[0]
        self.train_x, self.train_y = self.train_x[idx_train], self.train_y[idx_train]
        # Filter 3's and 8's from testing data
        idx_test = where((self.test_y == 3)|(self.test_y == 8))[0]
        self.test_x, self.test_y = self.test_x[idx_test], self.test_y[idx_test]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='svm classifier arguments')
    parser.add_argument('--kernel', type=str, default="linear",
                        help="Specifies the kernel type to be used in the algorithm")
    parser.add_argument('--c', type=float, default=1.0,
                    help="Penalty parameter C of the error term")
    parser.add_argument('--plot', type=bool, default=False,
                        help="Plot few Support vectors")
    args = parser.parse_args()

    # Load data
    data = Numbers("../data/mnist.pkl.gz")

    # Auto Analysis uncomment following
    # for kernel in ['linear','rbf']:
    #     for parameter in [1,1.5,2,5,10,15]:
    #         classifier = svm.SVC(C=parameter, kernel=kernel)
    #         classifier.fit(data.train_x, data.train_y)
    #         accuracy = classifier.score(data.test_x, data.test_y)
    #         print("#Accuracy: {:2.4f} #Kernel: {:7s} #Regularization Parameter_C: {:02.1f}".format(100 * accuracy, kernel,
    #                                                                                        parameter))
    # Init svm classifier
    classifier = svm.SVC(C=args.c, kernel=args.kernel)
    # Fit training data
    classifier.fit(data.train_x, data.train_y)
    # Check accuracy of classifier with testing data
    accuracy = classifier.score(data.test_x,data.test_y)
    # print necessary stuff
    print("#Accuracy: {:2.4f} #Kernel: {} #Regularization Parameter_C: {:1.0f}".format(100*accuracy,args.kernel,args.c))

    # if plot than plot top 5 support vector
    if args.plot:
        for i in range(1,11):
            plt.subplot(2,5, i)
            if i < 6:
                # top starting vectors will be 3's
                plt.title('Support Vector:{}'.format(3),fontsize=8)
                plt.imshow(classifier.support_vectors_[i-1].reshape(28, 28), cmap=cm.gray)
            else:
                # top ending vectors will be 8's
                plt.title('Support Vector:{}'.format(8),fontsize=8)
                plt.imshow(classifier.support_vectors_[5-i].reshape(28, 28), cmap=cm.gray)
        plt.show()