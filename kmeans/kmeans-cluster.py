#-*- coding: utf-8 -*-
#!/usr/bin/python
#
# usage
#   $ python kmeans-cluster.py <cluster number> <data filename> <columns> <centroid filename> <result filename>
#   $ python kmeans-cluster.py 3 iris.data "0,1,2,3" iris-centroid.csv result.txt
#   $ python kmeans-cluster.py 3 wine.data "1,2,3,4,5,6,7,8,9,10,11,12,13" wine-centroid.csv wine.out.txt

import sys
import csv
from pylab import plot,show
from numpy import vstack,array
from scipy.cluster.vq import kmeans,vq
from numpy import genfromtxt
import numpy as np



def readData(filename, columns):
    csv = genfromtxt(filename, delimiter=",", usecols=(map(int, columns.split(","))))
    return csv


def writeResult(filename, idx):
    w = open(filename, 'w', encoding="utf-8")
    
    for i in range(len(idx)):
        w.write("{}\n".format(idx[i]))
    w.close()


if __name__ == "__main__":

    k = sys.argv[1]
    data_filename = sys.argv[2]
    columns = sys.argv[3]
    centroid_filename = sys.argv[4]
    out_filename = sys.argv[5]
 
    read_data = readData(data_filename, columns)
    
    centroids,_ = kmeans(read_data, int(k))
    
    # assign each sample to a cluster
    idx,_ = vq(read_data,centroids)
    writeResult(out_filename, idx)

    np.savetxt(centroid_filename, centroids, delimiter=",")
    
    # plot result
    for i in range(int(k)):
        plot(read_data[idx==i, 0], read_data[idx==i, 1], 'o', markersize=3)
        plot(centroids[i:,0], centroids[i:,1],'^r', markersize=10)

    show()
