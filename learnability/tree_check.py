import multiprocessing
from itertools import combinations
from operator import itemgetter
from bst import BST
import time
from numpy import array, dot, unique, arctan2, spacing, single, tan, sort, random ,ndarray

def wrapper(args):
    obj,dataset,rademacher = args
    return obj.correlation(dataset,rademacher)

def estimate(dataset, hypothesis_generator, num_samples=500,
                        random_seed=0):
    sum = 0.0
    # cpu = multiprocessing.cpu_count()
    # print("Number of Core found:{}".format(cpu))
    # if cpu > 2:
    #     pool = multiprocessing.Pool(cpu - 2)
    # else:
    #     pool = multiprocessing.Pool(2)
    for ii in range(num_samples):
        if random_seed != 0:
            rademacher = coin_tosses(len(dataset), random_seed + ii)
        else:
            rademacher = coin_tosses(len(dataset))
        # DONE: complete this function
        # hypothesis_set = hypothesis_generator(dataset)
        # if cpu > 2:
        #     pool = multiprocessing.Pool(cpu - 2)
        # else:
        #     pool = multiprocessing.Pool(2)
        # result = pool.map(wrapper,((each,dataset,rademacher) for each in hypothesis_set))
        # pool.close()
        # pool.join()
        # sum += max(result)
        sum += max([each.correlation(dataset,rademacher)
                    for each in hypothesis_set])
    return sum/num_samples

def worker(combination):
    hypotheses_set = list()
    for each in combinations(dataset, combination):
        point = (min(each, key=itemgetter(0))[0], min(each, key=itemgetter(1))[1],
                 max(each, key=itemgetter(0))[0], max(each, key=itemgetter(1))[1])

        x = set([points.key for points in treeX.range((point[0], point[1]), (point[2], point[3]))])
        y = set([(points.key[1], points.key[0]) for points in treeY.range((point[1], point[0]), (point[3], point[2]))])

        if len(x & y) == combination: hypotheses_set.append((point[0], point[1],point[2], point[3]))
    return hypotheses_set

# def axis_aligned_hypotheses(dataset1):
#
#     global dataset = dataset1
#     [(treeX.insert((point[0],point[1])),treeY.insert((point[1],point[0])))
#      for point in dataset]
#
#     hypotheses_set = [[( float('inf'), float('inf'),float('inf'), float('inf'))]]
#
#     for combination in range(1,len(dataset)+1):
#         hypotheses_set = worker(combination, dataset, treeX, treeY)
#
#     for each in hypotheses_set:
#         for point in each:
#             yield AxisAlignedRectangle(point[0], point[1],point[2], point[3])



if __name__ == "__main__":
    # cpu = multiprocessing.cpu_count()
    # print("Number of Core found:{}".format(cpu))
    # pool = multiprocessing.Pool(4)
    pts = 50 * random.random((16, 2))
    t = time.perf_counter()
    print("Rademacher correlation of extra plane classifier %f" %
          estimate(pts, axis_aligned_hypotheses, num_samples=1))
    print(time.perf_counter() - t)