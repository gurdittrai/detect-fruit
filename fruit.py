import sys
import ntpath

#image
import numpy as np
import cv2
from matplotlib import pyplot as plt

from itertools import product

def example(r):
    #gray values
    gvalues = np.unique(r)

    #d=1 and q=horizontal or vertical
    h = np.zeros_like(r)
    v = np.zeros_like(r)

    for p in product(gvalues,gvalues):
        #horizontal
        for i in xrange(1,r.shape[0]+1):
            #avoid edges
            j = i-1 if ((i-1)>=0) else 0
            h[p[0],p[1]] += pairs(p,r[j:i])

        #vertical
        for i in xrange(1,r.shape[1]+1):
            #avoid edges
            j = i-1 if ((i-1)>=0) else 0
            v[p[0],p[1]] += pairs(p,r[:,i-1])
    print(h)
    print(v)

# "On-tree fruit recognition using texture properties and fruit"
# appendix 7
def pairs(p,arr):
    #count number of pairs
    count = 0
    #add useless columns for pair check
    arr = np.append(-1,arr)
    arr = np.append(arr,-1)
    for i in xrange(1,arr.shape[0]-1):
        #match adj pairs
        if arr[i] == p[0]:
            if arr[i-1] == p[1]:
                count += 1
            if arr[i+1] == p[1]:
                count += 1
    return count


#example
R = np.asarray([[0,0,1,1],[0,0,1,1],[0,2,2,2],[2,2,3,3]])
example(R)
