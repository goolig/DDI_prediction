import time

import networkx as nx
import random
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from d2d_evaluation import create_train_test_split_relese, create_train_test_split_ratio


#############to repeat the expermints:
import os
from keras import backend as K
import tensorflow as tf
#seed = 8
from utils import pickle_object, unpickle_object

seed = 123456#10 OK, 1234 better than xgboost,
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(seed)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
########################################
import multiprocessing.dummy as mp


class k():


    def __init__(self):
        self.G=None
        self.edges=None
        self.f_name='katz_middle_calc'

    def katz_similarity(self, t):
        i, j = t[0], t[1]
        l = 1
        neighbors = self.Graph[i]
        score = 0.0
        while l <= self.maxl:
            numberOfPaths = neighbors[0, j]  # neighbors.count(j)
            if numberOfPaths > 0:
                score += (self.beta ** l) * numberOfPaths
            l += 1
            if l <= self.maxl:
                neighborsForNextLoop = csr_matrix((1, self.G.number_of_nodes()), dtype=np.uint16)
                for k in neighbors.nonzero()[1]:
                    neighborsForNextLoop += (neighbors[0, k] * self.Graph[k])
                neighbors = neighborsForNextLoop
        self.katz_scores.append((i, j, score))
        self.done.add((i,j))
        if i % 10 == 0 and j % 100 == 0:
            print(i, j)
            start_time = time.perf_counter()
            # your code

            pickle_object(self.f_name+'matrix.p',self.katz_scores)
            elapsed_time =  time.perf_counter() - start_time
            print('left ',len(self.edges) - len(self.katz_scores),'edges')
            print('it took in seconds,', elapsed_time)
        # return score




    # Implementation of Katz's algorithm
    def katz(self):
        try:
            self.katz_scores = unpickle_object(self.f_name+'matrix.p')
        except:
            self.katz_scores = []
        self.done = set([ (x[0],x[1]) for x in self.katz_scores])

        print('read from file:', len(self.done))
        self.maxl = 3  # Number of iterations for Katz Algorithm (beta^maxl ~ 0)
        self.beta = 0.1  # The damping factor for Katz Algorithm
        # predictions = np.zeros((self.m.shape[0], self.m.shape[1]))
        # count = 0
        Graph = {}
        for n in self.G.nodes:
            Graph[n] = lil_matrix((1, self.G.number_of_nodes()), dtype=np.uint16)
            for node in list(self.G[n]):
                Graph[n][0, node] = 1
            Graph[n] = Graph[n].tocsr()
        print(f'working on {len(self.edges)} edges')
        # start_time = time.time()
        self.Graph = Graph
        p = mp.Pool(4)
        self.edges = self.edges - self.done
        p.map(self.katz_similarity, self.edges)  # range(0,1000) if you want to replicate your example

        p.close()
        p.join()

        return self.katz_scores

train_ratio=0.7
validation_ratio=0.0
test_ratio =0.3

#Holdout:
evaluation_method =['Retrospective', 'Holdout'][1]
new_version="5.1.1"
old_version = "5.0.0"

#spliting to train\test
if evaluation_method == 'Retrospective':
    m_test,m_train,evaluator,test_tuples, i2d,evaluation_type,drug_id_to_name  = create_train_test_split_relese(old_relese = old_version,new_relese=new_version)
else:
    m_test,m_train,evaluator,test_tuples, i2d, evaluation_type,drug_id_to_name = create_train_test_split_ratio(new_version,train_ratio,validation_ratio,test_ratio)

G = nx.from_numpy_matrix(m_train)
edges = sorted([(i, j) for i in G.nodes() for j in G.nodes() if j>=i])
def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]

edges_parts = split_list(edges,wanted_parts=10)
part = 2 # done: 0, 1, 4,3,5  Working: here: 5, VPN: .
katz_calc = k()
katz_calc.G = G
katz_calc.edges = set(edges_parts[part])
katz_scores = katz_calc.katz()

pickle_object(f'kats_scores_final_{part}.pickle',katz_scores)