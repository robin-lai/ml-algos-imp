
from random import choice,sample
import numpy as np
# use three way to implement euclidean
def euclidean1(x1,x2):
    ret = 0.0
    for i in range(len(x1)):
        ret += x1[i]*x1[i] + x2[i]*x2[i]
    return ret

def euclidean2(x1,x2):
    c = [x1[i]-x2[i] for i in range(len(x1))]
    c_square = [i*i for i in c]
    return sum(c_square)

def euclidean3(x1,x2):
    return sum(np.square(np.array(x1)-np.array(x2)))

"""
data is nest list
"""
def kmeans(k, data, epsilon):
    n = len(data)

    centers = []
    clusters = {}
    for c in sample(data,k):
        centers.append(c)
    for i in range(len(centers)):
        print('the random centers are:',i,':',centers[i],end=' ')
        print()
    #数据结构1：以簇心为key,簇内其它点为value列表中的点。
    #数据结构2：簇心用一个nest list表示，不用dict.
    #初始化clusters,因为后面要进行append操作
    for i in range(k):
        clusters[i]=[]

    while True:
        # for ele in data:
        #     best_center = -1
        #     distance = 99999999
        #     for i in range(len(centers)):
        #         current_dist = euclidean(ele,centers[i])
        #         if current_dist<distance :
        #             best_center = i
        #             distance = current_dist
        #     clusters[best_center].append(ele)

        for p_index in range(len(data)):
            nearest_neigh = -1
            distance = np.inf
            for c_index in range(len(centers)):
                current_dist = euclidean1(centers[c_index],data[p_index])
                if current_dist<distance:
                    best_center = c_index
                    distance = current_dist
            clusters[best_center].append(p_index)
            #clusters[best_center].append(i)

        for k,v in clusters.items():
            print('the points in cluster:',k,'are:',v,end=' ')
            print()
        #calculate mean of clusters
        new_centers = []
        for p_indexs in clusters.values():
            points = [data[p_index] for p_index in p_indexs]
            mean = np.sum(points,axis=0)/len(points)
            new_centers.append(mean)

        print('the new centers are:',new_centers,end=' ')
        #break

        #判断是否结束
        diff = .0
        for c in range(k):
            diff += euclidean3(new_centers[c],centers[c])
        if diff<epsilon:
            break
        centers = new_centers








kmeans(3,[[k,k] for k in range(100)],epsilon=2.0)