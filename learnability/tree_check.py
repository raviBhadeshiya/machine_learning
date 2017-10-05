from itertools import combinations
from operator import itemgetter
from random import randint, seed

from bst import BST
from numpy import array, dot, unique, arctan2, spacing, single, tan, sort
dataset = [(1., 1.), (2., 2.),(3.,0.),(4., 2.)]

tree = BST()
[tree.insert(point) for point in dataset]

print(tree)
# Generate list of points in each range (x,y)
s = spacing(single(1))
for r in tree.range((5.0,1.0),(4.0,2.0)):
    print(r.key)

