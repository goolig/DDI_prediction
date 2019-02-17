import multiprocessing.dummy as mp
import os

import networkx as nx
import numpy
import pandas as pd
import numpy as np
import math
from scipy.sparse import csr_matrix, lil_matrix
import random
from utils import unpickle_object, pickle_object

class graph_features_creator():
    def __init__(self,mat,maxl = 2, beta = 0.1,G=None):
        self.file_name_pickle = self.get_pickle_file_name(mat)
        if G==None:
            self.G = nx.from_numpy_matrix(mat)
        else: self.G=G
        self.edges = [(i, j) for i in self.G.nodes() for j in self.G.nodes() if j>=i]
        self.maxl = maxl  # Number of iterations for Katz Algorithm (beta^maxl ~ 0)
        self.beta = beta  # The damping factor for Katz Algorithm

    def get_pickle_file_name(self,mat):
        value_to_hash = len(mat.nonzero()[0]) + mat.shape[0] + mat.shape[1] #raises the probablity for collisions, but should do the job for a small number of elements
        return os.path.join('pickles',"graph_features_" + str(hash(value_to_hash)) + ".pickle")

    feature_names = ['Jaccard coefficient', 'Average common neighbors','Shortest Path Length', 'Average Jaccard coefficient', 'Cosine', 'Community','Adamic Adar','Katz','Preferential Attachment']

    def create_graph_features(self):
        # jaccard
        features = pd.DataFrame(nx.jaccard_coefficient(self.G, self.edges))
        features.set_index([features.columns[0], features.columns[1]], inplace=True)
        features.rename(columns={2: "jaccard"}, inplace=True)
        print('done jaccard')
        # commmon friends level 2
        common_friends_dict = {}
        for v1,v2 in self.edges:
            result = len(list(nx.common_neighbors(self.G, v1, v2)))
            common_friends_dict[(v1, v2)] = result
            common_friends_dict[(v2, v1)] = result
        features["number_of_nig"] = None
        for v1,v2 in self.edges:
            total_friends1 = []
            for z in self.G[v2]:
                total_friends1.append(common_friends_dict[v1, z])
            total_friends2 = []
            for z in self.G[v1]:
                total_friends2.append(common_friends_dict[v2, z])
            features.at[(v1, v2), "number_of_nig"] = np.mean(np.nan_to_num([np.mean(total_friends1),np.mean(total_friends2)]))
        print('done level 2 neig')
        # shortest path
        features["shortest_path_len"] = None
        for v1,v2 in self.edges:
            try:
                features.at[(v1, v2), "shortest_path_len"] = nx.shortest_path_length(self.G, v1, v2)#TODO: calculate all at once
            except:
                features.at[(v1, v2), "shortest_path_len"] = 20 #not path between the two nodes. this value should be larger than max length.
        print('done shortest path')
        # jaccard level 2
        features["jaccard_level2"] = None
        jac_dict = {}
        for v1,v2 in self.edges:
            jaccard_score = features.at[(v1, v2), "jaccard"]
            jac_dict[(v1, v2)] = jaccard_score
            jac_dict[(v2, v1)] = jaccard_score
        for v1, v2 in self.edges:
            jaccard_score2_v1 = []
            for common_neighbors in self.G[v2]:
                jaccard_score2_v1.append(jac_dict[(v1, common_neighbors)])
            jaccard_score2_v2 = []
            for common_neighbors in self.G[v1]:
                jaccard_score2_v2.append(jac_dict[(v2, common_neighbors)])
            #assert len(jaccard_score2_v1)>0 and len(jaccard_score2_v2)>0, "mean with len=0 returns nan"
            features.at[(v1, v2), "jaccard_level2"] = np.mean(np.nan_to_num([np.mean(jaccard_score2_v1),np.mean(jaccard_score2_v2)]))
        print('done jaccard level 2')
        # cosine
        features["cosine"] = None
        for v1,v2 in self.edges:
            p_a = list(nx.preferential_attachment(self.G, [(v1, v2)]))[0][2] #TODO: calculate all at once.
            if p_a != 0:
                features.at[(v1, v2), "cosine"] = len(list(nx.common_neighbors(self.G, v1, v2))) / p_a
            else:
                features.at[(v1, v2), "cosine"] = 0
        print('done cosine')
        # is same community
        #coms = community.best_partition(self.G)
        features["is_same_community"] = None
        for v1, v2 in self.edges:
            result = int(random.getrandbits(1))
            #if coms[v1] == coms[v2]:
            #    result = 1
            features.at[(v1, v2), "is_same_community"] = result
        print('done same community')
        # adamic adar:
        features["adamic_adar"] = None
        for v1,v2 in self.edges:
            preds = []
            common_neighbors = nx.common_neighbors(self.G, v1, v2)
            for v3 in common_neighbors:
                if math.log(self.G.degree(v3)) > 0:
                    preds.append(1.0 / math.log(self.G.degree(v3)))
                else:
                    preds.append(0)
            features.at[(v1, v2), "adamic_adar"] = sum(preds) #networkx has a bug when a node has 1 neighbor (with itself)
        print('done adamic adar')
        features["Katz"] = None
        katz_preds = self.katz()
        for v1,v2,pred in katz_preds:
            features.at[(v1, v2), "Katz"] = pred
        print('done Katz')
        features["preferential_attachment"] = None
        pa_preds = nx.preferential_attachment(self.G, self.edges)
        for v1,v2,pred in pa_preds:
            features.at[(v1, v2), "preferential_attachment"] = pred
        print('done Prefential attachment')
        return features

    def normalize_features(self,features):
        return (features - features.mean()) / (features.std())

    def convert_features_to_dict(self,features):
        print('converting to dict')
        features_dict = {}  # .loc[466,466].values
        for v1, v2 in self.edges:
            features_dict[v1 ,v2] = features.loc[v1, v2].values
        return features_dict

    def get_normalized_feature_dict(self):
        try:
            return unpickle_object(self.file_name_pickle)
        except:
            print("can't unpickle features, calculating")
            result = self.convert_features_to_dict(self.normalize_features(self.create_graph_features()))
            pickle_object(self.file_name_pickle, result)
            return result

    def katz_similarity(self, t):
        i, j = t[0],t[1]
        l = 1
        neighbors = self.Graph[i]
        score = 0.0
        while l <= self.maxl:
            numberOfPaths = neighbors[0,j]#neighbors.count(j)
            if numberOfPaths > 0:
                score += (self.beta ** l) * numberOfPaths
            l += 1
            if l <= self.maxl:
                neighborsForNextLoop = csr_matrix((1, self.G.number_of_nodes()), dtype=np.uint16)
                for k in neighbors.nonzero()[1]:
                    neighborsForNextLoop += (neighbors[0,k]*self.Graph[k])
                neighbors = neighborsForNextLoop
                self.katz_scores.append((i, j,score))
        if i%10==0 and j% 100==0:
            print(i,j)

    # Implementation of Katz's algorithm
    def katz(self):
        # predictions = []
        # Graph = {}
        # for n in self.G.nodes:
        #     Graph[n] = lil_matrix((1, self.G.number_of_nodes()), dtype=np.uint16)
        #     for node in list(self.G[n]):
        #         Graph[n][0,node]=1
        #     Graph[n]=Graph[n].tocsr()
        # print(f'working on {len(self.edges)} edges')
        # self.Graph = Graph
        # self.katz_scores=predictions
        # p = mp.Pool(4)
        # p.map(self.katz_similarity, self.edges)  # range(0,1000) if you want to replicate your example
        # p.close()
        # p.join()
        #return self.katz_scores
        arrays = []
        for i in range(10):
            part = unpickle_object(f'kats_scores_final_{i}.pickle')
            arrays.append(part)
        katz_scores = [(int(x[0]), int(x[1]), x[2]) for x in numpy.concatenate(arrays)]
        return katz_scores