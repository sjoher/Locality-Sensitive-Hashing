#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import numpy as np
import scipy.sparse as sps
from scipy.sparse import find
from scipy import sparse
import pandas as pd
from numpy import genfromtxt
import pprint, pickle
import time
import itertools
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
from joblib import Parallel, delayed
import argparse

def Jaccard_signature(signature_length, data):
    data = data.tocsr()
    permutations = np.array([np.random.permutation(data.shape[0]) for i in range(signature_length)]) # first find all permutations and 
                                                                                      # then use them as index in permuted_mat
    signatures = np.full((signature_length, data.shape[1]), np.inf)
    for perm in range(signature_length):
        permuted_mat = data[permutations[perm,:].argsort(),:] # maak een array met 1 row en aantal user kolommen
        i=0
        while(np.max(signatures[perm,:])==np.inf): # iterate over rows (movies)
            position = permuted_mat[i,:].nonzero()[1] # this gives all the users which have seen the ith movie (ith row)!!!
            signatures[perm, position] = np.minimum(i,signatures[perm, position]) # give the columns in the new matrix
                                                                                  # the value where they have a 1.
            i=i+1
    return(signatures)

def Cosine_signature(signature_length, data):
    hyperplanes = np.random.randn(data.shape[0], signature_length)
    signature_mat = np.sign(hyperplanes.T * data)
    return signature_mat

# find the pairs that fall into same bucket 
# inputs: signatureMatrix =signature matrix, rows = rows
# output: candidate_pairs = all combinatios of pairs into the same bucket
def LSH2(signatureMatrix, rows, func1, t_sign, func2, t, data, filename, time_limit):
    start = time.time()
    
    No_approved = 0
    No_candidates = 0
    filename = str(filename)+'.txt'
    f = open(filename, "w")
    data = data.tocsc()
    nrofUsers = np.shape(signatureMatrix)[1]   
    pairs = []
    
    #loop over bands
    for band in range(np.int(np.ceil(signatureMatrix.shape[0]/rows))):

        buckets = {}
        buckets_with_pairs = {}
        
        # create the bucket which is a dictionary with keys being the unique vectors of users and values a list 
        # with all the users Ids having the corresponding key
        # if two or more users have the same key, the key is saved in buckets_with_pairs. 
        # We are looping on the buckets_with_pairs for efficiency
        for user in range(nrofUsers):
            bucket_key = tuple(signatureMatrix[(band*rows):(band+1)*rows, user])
            if bucket_key in buckets:
                buckets[bucket_key].append(user)
                buckets_with_pairs[bucket_key] = 1
            else:
                buckets[bucket_key] = []
                buckets[bucket_key].append(user)
        #make all users in every bucket into pairs of two and return everything
        for bucket in buckets_with_pairs.keys():            
            combs_b = list(map(tuple, itertools.combinations(buckets[bucket],2))) 
            for j in combs_b: 
                if (time.time()-start)>time_limit:
                    return(pairs,No_approved,No_candidates)
                No_candidates += 1
                signature_sim = func1(signatureMatrix[:,j[0]],signatureMatrix[:,j[1]]) # find signature similarity
                if signature_sim>t_sign and tuple(sorted(j)) not in pairs: #to avoid dublicates
                    No_approved += 1
                    sim = func2(np.squeeze(data[:,j[0]].toarray()),np.squeeze(data[:,j[1]].toarray())) # now find actual similarity
                    if sim>t:
                        pair = np.sort(np.array(j))
                        f = open(filename, 'a')
                        np.savetxt(f, pair[np.newaxis], fmt = "%s",delimiter=",")
                        f.close()
                        pairs.append(tuple(sorted(j)))
                        
    return(pairs,No_approved,No_candidates)

#find the signature similarity of many pairs with the similarity method as input
def signature_similarity(candidate_pairs,func,signatureMatrix):
    similarities = np.zeros(len(candidate_pairs))
    for index,j in enumerate(candidate_pairs):
        similarities[index] = func(signatureMatrix[:,j[0]],signatureMatrix[:,j[1]])
    return(similarities)

#find the Jaccard similarity for one pair from signature matrix
def jaccard_signature_sim(user1,user2):
    return(np.mean(user1==user2))


#find the Jaccard similarity for one pair from sparse matrix
def jaccard_sim(user1,user2):
    numerator = sum(user1*user2)
    denominator = len((user1+user2).nonzero()[0])
    return(numerator/denominator)

#find the cosine_sim for one pair
def cosine_sim(user1,user2):
    num = np.dot(user1,user2)
    denom = np.sqrt(user1.dot(user1))*np.sqrt(user2.dot(user2))
    theta = np.arccos(num/denom)
    sim = 1-(math.degrees(theta)/180)
    return sim

#find the real similarity of many pairs with the similarity method as input
def similarity(pairs,data,func2):
    data = data.tocsc() # make sure the sparse matrix is in Compressed Sparse Column form
    output = np.zeros(len(pairs))
    for index,i in enumerate(pairs):
        output[index] = func2(np.squeeze(data[:,i[0]].toarray()),np.squeeze(data[:,i[1]].toarray()))
    return(output)


parser = argparse.ArgumentParser(description = "Similar users")
parser.add_argument('-d', '--datapath', type = str, help = 'Path to file')
parser.add_argument('-s', '--seed', type = int, help = 'Random seed number')
parser.add_argument('-m', '--method', type = str, help = 'Similarity measure, js, cs or dcs')
args = parser.parse_args()


if __name__ == '__main__':
    d = args.datapath
    s = args.seed
    m = args.method
    
    np.random.seed(s)
    path = d # d is het path in de assignment
    netflix_dat=np.load(path)
    sparse_mat = sps.coo_matrix((netflix_dat[:,2], (netflix_dat[:,1], netflix_dat[:,0])))
    sparse_mat = sparse_mat.tocsc()
    sparse_mat = sparse_mat[1:,1:]
    netflix_dat[:,2] = 1 
    sparse_mat_bool = sps.coo_matrix((netflix_dat[:,2], (netflix_dat[:,1], netflix_dat[:,0])))
    sparse_mat_bool = sparse_mat_bool.tocsr()
    sparse_mat_bool = sparse_mat_bool[1:,1:]

    if m == 'js':
        signature_length = 150
        row = 5
        t = 0.5
        t_sign = 0.48
        data = sparse_mat_bool
        func1 = jaccard_signature_sim
        func2 = jaccard_sim
        signatureMatrix = Jaccard_signature(signature_length, data)
        time_limit = 1550
        a,b,c = LSH2(signatureMatrix,row, func1, t_sign, func2, t, data, m, time_limit)
    elif m == 'cs':
        signature_length = 210
        row = 14
        t = 0.73
        t_sign = 0.66
        data = sparse_mat
        func1 = cosine_sim
        func2 = cosine_sim 
        signatureMatrix = Cosine_signature(signature_length, data)
        time_limit = 1600
        a,b,c = LSH2(signatureMatrix,row, func1, t_sign, func2, t, data, m, time_limit)
    elif m == 'dcs':
        signature_length = 210
        row = 14
        t = 0.73
        t_sign = 0.66
        data = sparse_mat_bool
        func1 = cosine_sim
        func2 = cosine_sim 
        signatureMatrix = Cosine_signature(signature_length, data)
        time_limit = 1600
        a,b,c = LSH2(signatureMatrix,row, func1, t_sign, func2, t, data, m, time_limit)
    
