__author__ = "Ravi Bhadeshiya"
__email__ = "ravib@terpmail.umd.edu"

from itertools import combinations
from operator import itemgetter
from random import randint, seed

import time
from bst import BST
from numpy import array, dot, unique, arctan2, spacing, single, tan, sort, ndarray, random

kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]


class Classifier:
    def classify(self, point):
        raise NotImplementedError()

    def correlation(self, data, labels):
        """
        Return the correlation between a label assignment and the predictions of
        the classifier

        Args:
          data: A list of datapoints
          labels: The list of labels we correlate against (+1 / -1)
        """

        assert len(data) == len(labels), \
            "Data and labels must be the same size %i vs %i" % \
            (len(data), len(labels))

        assert all(x == 1 or x == -1 for x in labels), "Labels must be binary"

        prediction = [2 * self.classify(itr) - 1 for itr in data]
        return dot(labels, prediction) / float(len(labels))


class PlaneHypothesis(Classifier):
    """
    A class that represents a decision boundary.
    """

    def __init__(self, x, y, b):
        """
        Provide the definition of the decision boundary's normal vector

        Args:
          x: First dimension
          y: Second dimension
          b: Bias term
        """
        self._vector = array([x, y])
        self._bias = b

    def __call__(self, point):
        return self._vector.dot(point) - self._bias

    def classify(self, point):
        return self(point) >= 0

    def __str__(self):
        return "x: x_0 * %0.2f + x_1 * %0.2f >= %f" % \
               (self._vector[0], self._vector[1], self._bias)


class OriginPlaneHypothesis(PlaneHypothesis):
    """
    A class that represents a decision boundary that must pass through the
    origin.
    """

    def __init__(self, x, y):
        """
        Create a decision boundary by specifying the normal vector to the
        decision plane.

        Args:
          x: First dimension
          y: Second dimension
        """
        PlaneHypothesis.__init__(self, x, y, 0)


class AxisAlignedRectangle(Classifier):
    """
    A class that represents a hypothesis where everything within a rectangle
    (inclusive of the boundary) is positive and everything else is negative.

    """
    def __init__(self, start_x, start_y, end_x, end_y):
        """

        Create an axis-aligned rectangle classifier.  Returns true for any
        points inside the rectangle (including the boundary)

        Args:
          start_x: Left position
          start_y: Bottom position
          end_x: Right position
          end_y: Top position
        """
        assert end_x >= start_x, "Cannot have negative length (%f vs. %f)" % \
                                 (end_x, start_x)
        assert end_y >= start_y, "Cannot have negative height (%f vs. %f)" % \
                                 (end_y, start_y)

        self._x1 = start_x
        self._y1 = start_y
        self._x2 = end_x
        self._y2 = end_y

    def classify(self, point):
        """
        Classify a data point

        Args:
          point: The point to classify
        """
        return (self._x1 <= point[0] <= self._x2) and \
               (self._y1 <= point[1] <= self._y2)

    def __str__(self):
        return "(%0.2f, %0.2f) -> (%0.2f, %0.2f)" % \
               (self._x1, self._y1, self._x2, self._y2)


class ConstantClassifier(Classifier):
    """
    A classifier that always returns true
    """

    def classify(self, point):
        return True


def constant_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over the single constant
    hypothesis possible.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    yield ConstantClassifier()


def origin_plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector.  The classification decision is
    the sign of the dot product between an input point and the classifier.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    # DONE: Complete this function
    if not isinstance(dataset, ndarray):
        dataset = array(dataset)

    unique_angles = unique(sort(arctan2(dataset[:, 1], dataset[:, 0])))

    mean_slopes = [tan((unique_angles[itr + 1] + unique_angles[itr]) / 2.0)
                   for itr in range(len(unique_angles) - 1)
                   ]
    mean_slopes.append(tan(unique_angles[-1] + spacing(single(1))))  # To handle extreme case
    # hypothesis_set = [[[each, -1],[-each, 1]] for each in mean_slopes]
    #
    # for each in hypothesis_set:
    #     for one in each:
    #         yield OriginPlaneHypothesis(one[0], one[1])
    for each in mean_slopes:
        for one in [[each, -1], [-each, 1]]:
            yield OriginPlaneHypothesis(one[0], one[1])


def plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector and a bias.  The classification
    decision is the sign of the dot product between an input point and the
    classifier plus a bias.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    # Complete this for extra credit
    if isinstance(dataset, ndarray):
        data = dataset.tolist()[:]
    else:
        data = dataset[:]  # copy data

    while len(data) != 1:  # Loop over each and every data
        new_origin = data.pop()  # Pop the each point from data
        # Trick is to shift data to that point as origin
        shifted_data = array(dataset)[:len(data)] - array([new_origin] * len(data))
        # And Compute the origin plane hypotheses for that shifted data
        unique_angles = unique(sort(arctan2(shifted_data[:, 1], shifted_data[:, 0])))

        mean_slopes = [tan((unique_angles[itr + 1] + unique_angles[itr]) / 2.0)
                       for itr in range(len(unique_angles) - 1)
                       ]
        mean_slopes.append(tan(unique_angles[-1] + spacing(single(1))))  # To handle extreme case
        # y - new_originY = m(x-new_originX) + 0
        #               y = m*x - m*new_originX + new_originY
        #               y = m*x + c ; c = new_originY - m*new_originX
        for each_slope in mean_slopes:
            for one in [[each_slope, -1, new_origin[1] - each_slope * new_origin[0]],
                        [-each_slope, 1, each_slope * new_origin[0] - new_origin[1]]]:
                yield PlaneHypothesis(one[0], one[1], one[2])


def axis_aligned_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are axis-aligned rectangles

    Args:
      dataset: The dataset to use to generate hypotheses
    """
    # DONE: complete this function
    treeX = BST()
    treeY = BST()
    [(treeX.insert((point[0], point[1])), treeY.insert((point[1], point[0])))
     for point in dataset]

    hypotheses_set = [AxisAlignedRectangle(float('inf'), float('inf'),
                                           float('inf'), float('inf'))]

    for combination in range(1, len(dataset) + 1):
        for each in combinations(dataset, combination):
            point = (min(each, key=itemgetter(0))[0], min(each, key=itemgetter(1))[1],
                     max(each, key=itemgetter(0))[0], max(each, key=itemgetter(1))[1])

            x = set([points.key for points in treeX.range((point[0], point[1]), (point[2], point[3]))])
            y = set(
                [(points.key[1], points.key[0]) for points in treeY.range((point[1], point[0]), (point[3], point[2]))])
            # count= [1 for points in tree.range((each._x1,each._y1),(each._x2,each._y2))
            #          if each.classify(points.key)]
            if len(x & y) == combination: hypotheses_set.append(AxisAlignedRectangle(point[0], point[1],
                                                                                     point[2], point[3]))
            # rectangles = [(min(each, key=itemgetter(0))[0], min(each, key=itemgetter(1))[1],
            #                max(each, key=itemgetter(0))[0], max(each, key=itemgetter(1))[1])
            #               for each in combinations_set]

            # for each in rectangles:
            #
            #     x = set([points.key for points in treeX.range((each[0],each[1]),(each[2],each[3]))])
            #     y = set([(points.key[1],points.key[0]) for points in treeY.range((each[1], each[0]), (each[3], each[2]))])
            #     # count= [1 for points in tree.range((each._x1,each._y1),(each._x2,each._y2))
            #     #          if each.classify(points.key)]
            #     if len(x&y) == itr: hypotheses_set.append(AxisAlignedRectangle( each[0],each[1],
            #                                                                     each[2],each[3])) # Check validity

    for each in hypotheses_set:
        yield each


def coin_tosses(number, random_seed=0):
    """
    Generate a desired number of coin tosses with +1/-1 outcomes.

    Args:
      number: The number of coin tosses to perform

      random_seed: The random seed to use
    """
    if random_seed != 0:
        seed(random_seed)

    return [randint(0, 1) * 2 - 1 for x in range(number)]


def rademacher_estimate(dataset, hypothesis_generator, num_samples=500,
                        random_seed=0):
    """
    Given a dataset, estimate the rademacher complexity

    Args:
      dataset: a sequence of examples that can be handled by the hypotheses
      generated by the hypothesis_generator

      hypothesis_generator: a function that generates an iterator over
      hypotheses given a dataset

      num_samples: the number of samples to use in estimating the Rademacher
      correlation
    """
    hypothesis_set = hypothesis_generator(dataset)
    hypothesis_set= [each for each in hypothesis_set]
    sum = 0.0
    for ii in range(num_samples):
        if random_seed != 0:
            rademacher = coin_tosses(len(dataset), random_seed + ii)
        else:
            rademacher = coin_tosses(len(dataset))
        # DONE: complete this function
        sum += max([each.correlation(dataset, rademacher)
                    for each in hypothesis_set])
    return sum / num_samples


if __name__ == "__main__":
    print("Rademacher correlation of constant classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, constant_hypotheses))
    print("Rademacher correlation of rectangle classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, axis_aligned_hypotheses))
    print("Rademacher correlation of plane classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, origin_plane_hypotheses))
    print("Rademacher correlation of extra plane classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, plane_hypotheses))

    # pts = 100 * random.random((12, 2))
    # #
    #
    # t = time.process_time()
    # print("Rademacher correlation of plane classifier %f" %
    #       rademacher_estimate(pts, origin_plane_hypotheses))
    # print(time.process_time() - t)
    #
    # t = time.process_time()
    # print("Rademacher correlation of extra plane classifier %f" %
    #       rademacher_estimate(pts, plane_hypotheses))
    # print(time.process_time() - t)
    # #
    # t = time.process_time()
    # print("Rademacher correlation of rectangle classifier %f" %
    #       rademacher_estimate(pts, axis_aligned_hypotheses))
    # print(time.process_time() - t)
