import keras.backend as K
import numpy
import pandas as pd
import random
import networkx as nx
#import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Embedding, Flatten, Add, Conv2D, Reshape, Conv1D
import xgboost as xgb

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate, Multiply
from keras.optimizers import Adam
from keras import regularizers
from keras.initializers import RandomNormal

from keras.models import load_model
from keras.optimizers import SGD
from keras.regularizers import l2
from scipy.sparse.linalg import svds
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances, roc_auc_score

from abc import ABC, abstractmethod

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from xgboost import DMatrix

from d2d_evaluation import drug_evaluator, create_train_validation_split, \
    create_train_validation_split_single_sample_per_drug, create_train_validation_split_rectangle
from d2d_graph_features_creator import graph_features_creator
from utils import make_mat_sym
import pickle





class drugs_graph_predictor():
    algo_names = ['RAI','jaccard','adamic_adar_index','preferential_attachment']

    def __init__(self,m,edges_to_prediction,name):
        #super().__init__(m,f'{algo_name}')
        self.predictions = None
        self.m = m.copy()
        self.name=name
        self.edges_to_prediction = edges_to_prediction
        self.color = 'black'
        self.linestyle = '-.'
        self.linewidth = 1

    def fit(self):
        number_of_drugs = self.m.shape[0]
        edge_list = [(x, y) for x in range(number_of_drugs) for y in range(number_of_drugs) if
                     self.m[x, y] > 0 and x > y]
        self.G = nx.Graph()
        self.G.add_nodes_from(range(
            number_of_drugs))  # some edges are not added in train, then the prediction throws exception. This will prevent the exception but will give shitty results for those nodes.
        self.G.add_edges_from(edge_list)

    def predict(self,):
        preds = None
        if self.algo_name == 'RAI':
            print('RAI')  # 0.707
            preds = nx.resource_allocation_index(self.G, self.edges_to_prediction)
        if self.algo_name == 'jaccard':
            print('jaccard') # 0.628
            preds = nx.jaccard_coefficient(self.G, self.edges_to_prediction)
        if self.algo_name == 'adamic_adar_index':
            print('adamic_adar_index') #0.687
            preds = nx.adamic_adar_index(self.G, self.edges_to_prediction )
        if self.algo_name == 'preferential_attachment':
            print('preferential_attachment') #0.498
            preds = nx.preferential_attachment(self.G,self.edges_to_prediction )

        if preds is None:
            raise ValueError('Algorithm was not found: %s Or something weird happened in prediction' % self.algo_name)

        predictions1 = [(i,v) for (v, i) in sorted([(p, (u, v)) for (u, v, p) in preds],
                                               reverse=True)]  # get the cells of matrix in ascending order of cell value
        print(1)
        predictions2 = predictions1  # the following is redundent here... #[t for t in predictions1 if t[0]<t[1]] # just upper half of the matrix and predictions larger than 0
        return predictions2


class drugs_SVD_predictor():
    def __init__(self,m,k=100,show_graph=False):
        name=f'SVD {k} components'
        #super().__init__(m,)
        self.predictions = None
        self.m = m.copy()
        self.name=name
        self.k=k
        self.show_graph=show_graph

    def predict(self):
        """predicts links in descinding order. only half of the matrix"""
        print('predicting')
        predictions = [(i,v) for (v, i) in sorted([(v, i) for (i, v) in np.ndenumerate(self.lower_dim_m)] , reverse=True)] #get the cells of matrix in ascending order of cell value
        predictions = [(t,v) for t,v in predictions if t[0]>t[1] and self.m[t[1],t[0]] == 0 ] # just half of the matrix
        print('done predicting')
        return predictions

    def fit(self): #100 is best for random spit on latest data. 50 is best for release split on 5.0.2 vs 5.0.11
        print('calculating svd')
        #self.m = normalize(self.m, axis=1, norm='l2')
        # mean_overall = self.m.mean()
        # mean_rows = self.m.mean(1)
        # mean_cols = self.m.mean(0)
        u, s, vt = svds(self.m , k=self.k)
        if self.show_graph:
            print(list(reversed(s)))
            #plt.plot(list(reversed(s)))
            #plt.show(block=True)
        #self.m = dok_matrix(self.m)
        self.lower_dim_m = np.dot(u, np.dot(np.diag(s), vt))

        print('done calculating svd')


class drugs_xgboost_predictor():
    feature_names = [ 'Average common neighbors',
                      'Average Jaccard coefficient',
                      #'Shortest Path Length', 'Preferential Attachment',
                     'Adamic/Adar',
                     '$Katz_b$'
                      ]  # ,
    def __init__(self,m,test_tuples,colsample_bytree=1,n_estimators=100,random_state=1234,subsample=1,learning_rate=0.01,max_depth=3):
        #super().__init__()
        self.colsample_bytree=colsample_bytree
        self.n_estimators=n_estimators
        self.random_state=random_state
        self.subsample=subsample
        self.learning_rate=learning_rate
        self.max_depth=max_depth
        self.predictions = None
        self.m = m.copy()
        self.name='XGBoost'
        self.features=[]
        for fn in self.feature_names:
            self.features.append(graph_features_creator.feature_names.index(fn))
        self.mlp_pred=None
        self.gmf_pred =None
        self.test_tuples = test_tuples
        self.color = 'black'
        self.linestyle = '-'
        self.linewidth = 1
        features_creator = graph_features_creator(m)
        self.features_dict = features_creator.get_normalized_feature_dict()
        self.num_features = len(self.features_dict[(0, 0)])
        self.num_of_drugs = self.m.shape[0]
        self.tanimoto_pred=None
        self.nn_pred=None

    def fit(self):
        fd = self.features_dict
        keys_items = [(k,v[self.features]) for k, v in fd.items()]
        # if self.mlp_pred is not None:
        #     keys_items = [(k,np.append(v, [self.mlp_pred[k[1], k[0]]])) for k, v in keys_items] #adding the NN prediction
        # if self.gmf_pred is not None:
        #     keys_items = [(k,np.append(v, [self.gmf_pred[k[1], k[0]]])) for k, v in keys_items] #adding the NN prediction
        if self.nn_pred is not None:
            pickle_off = open(self.nn_pred ,"rb")
            nn_predicted_mat = pickle.load(pickle_off)
            pickle_off.close()
            keys_items = [(k,np.append(v, [nn_predicted_mat[k[0],k[1]]])) for k, v in keys_items] #adding the NN prediction #nn_predicted_mat[1],
        if self.tanimoto_pred is not None:
            pickle_off = open(self.tanimoto_pred ,"rb")
            tanimoto_predicted_mat = pickle.load(pickle_off)
            pickle_off.close()
            keys_items = [(k,np.append(v, [tanimoto_predicted_mat[k[0],k[1]]])) for k, v in keys_items] #adding the NN prediction #nn_predicted_mat[1],

        keys = [k for k, v in keys_items]
        keys_items = [np.append(v, [int(self.m[k[1], k[0]])]) for k, v in keys_items]
        df = pd.DataFrame(keys_items)
        X, Y = df[df.columns[0:-1]], df[df.columns[-1]]
        #sum = [x[0] for x in self.m.sum(axis=1).tolist()]
        #weights = [np.log((sum[x[0]] + sum[x[1]])) for x in keys]
        num_pos = int(len(self.m.nonzero()[0])/2)
        one_one_ratio =  ((self.num_of_drugs*(self.num_of_drugs-1))/2 - num_pos ) / (num_pos)
        #print(f'pos one_one_ratio {one_one_ratio}')

        params = { #300 trains in 1 hour on laptop
             #'min_child_weight': [100,500,1000,2000,5000],
             'learning_rate': [0.001,0.0001,0.00001],
             'subsample': [0.05,0.1,0.2],
             'colsample_bytree': [0.5,0.75,1],
             'colsample_bylevel':[0.5,0.75,1],
             #'max_depth':[2,4,6],
        #     #'n_estimators':[100]
         }

        #
        # gbm = xgb.XGBClassifier(n_estimators=100, learning_rate=0.01, colsample_bylevel=0.75,
        #
        #                          max_depth=5,min_child_weight=5000, #was 4 and 5000
        #                          ,,random_state=1234)#reg_alpha=0.1
        gbm = xgb.XGBClassifier(colsample_bytree=self.colsample_bytree,scale_pos_weight = one_one_ratio,
                                n_estimators=self.n_estimators,random_state=self.random_state,subsample=self.subsample,
                                learning_rate=self.learning_rate,max_depth=self.max_depth)  # For retrospective
        #gbm = xgb.XGBClassifier(colsample_bytree=0.5,scale_pos_weight = one_one_ratio,n_estimators=100,random_state=1234,learning_rate=0.05)  # For holdout

        gbm.fit(X,Y) #,eval_metric='auc'
        self.gbm=gbm
        self.X=X
        self.keys=keys

    def predict(self):
        prediction_scores = self.gbm.predict_proba(self.X)
        # folds = 2
        # skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
        # random_search = GridSearchCV(gbm, param_grid=params, scoring='roc_auc',
        #                                 cv=skf.split(X, Y), verbose=3)
        # random_search.fit(X, Y)
        # print(random_search.best_params_)
        # prediction_scores = random_search.predict_proba(X)
        # Import the model we are using
        # Instantiate model with 1000 decision trees
        #rf = RandomForestClassifier(n_estimators=10,class_weight='balanced',min_samples_split=100000)
        # Train the model on training data
        #rf.fit(X, Y)
        #prediction_scores = rf.predict_proba(X)


        print(f'predicted: {len(prediction_scores)}')
        predictions = np.zeros((self.m.shape[0], self.m.shape[1]))
        for i,score in enumerate(prediction_scores):
            predictions[self.keys[i][0], self.keys[i][1]] = score[1]
            predictions[self.keys[i][1], self.keys[i][0]] = score[1]
        self.predicted_mat = predictions
        s = set(self.test_tuples)
        predictions = [(i, v) for (v, i) in sorted([(v, i) for (i, v) in np.ndenumerate(predictions)],
                                                   reverse=True)]  # get the cells of matrix in ascending order of cell value
        print(1)
        predictions = [(t, v) for t, v in predictions
                       if (t[0], t[1]) in s]  # just half of the matrix and predictions larger than 0


        return predictions


class drugs_single_feature_predictor():
    def __init__(self,m,test_tuples,feature='',features_dict=None):
        #super().__init__(m,feature)

        self.predictions = None
        self.m = m.copy()
        self.name=feature

        if features_dict == None:
            features_creator = graph_features_creator(m)
            self.features_dict = features_creator.get_normalized_feature_dict()
        else:
            self.features_dict = features_dict
        self.num_features = len(self.features_dict[(0, 0)])
        self.num_of_drugs = self.m.shape[0]
        self.test_tuples = test_tuples

    def fit(self):
        pass

    def predict(self):
        print(f'working on {self.name}')
        index = graph_features_creator.feature_names.index(self.name)
        predictions = np.zeros((self.m.shape[0], self.m.shape[1]))
        for i in range(self.num_of_drugs):
            for j in range(0,i):
                value = self.features_dict[(j,i)][index]
                if self.name == 'Shortest Path Length':
                    value = 1/(1000+value)
                    assert value>0
                predictions[i, j] = value
                predictions[j, i] = value
        # s = set(self.test_tuples)
        # predictions = [(i, v) for (v, i) in sorted([(v, i) for (i, v) in np.ndenumerate(predictions)],
        #                                            reverse=True)]  # get the cells of matrix in ascending order of cell value
        # print(1)
        # predictions = [(t, v) for t, v in predictions
        #                if t[0] > t[1] and (((t[0], t[1]) in s) or (
        #             (t[1], t[0]) in s))]  # just half of the matrix and predictions larger than 0
        #
        s = set(self.test_tuples)
        predictions = [(i, v) for (v, i) in sorted([(v, i) for (i, v) in np.ndenumerate(predictions)],
                                                   reverse=True)]  # get the cells of matrix in ascending order of cell value
        print(1)
        predictions = [(t, v) for t, v in predictions
                       if (t[0], t[1]) in s]  # just half of the matrix and predictions larger than 0


        return predictions


class drugs_nn_predictor():
    def __init__(self, m, test_tuples, validation_tuples=None, validation_target=None,name='AMF',propagation_factor=None, mul_emb_size = 128, dropout=0.3, epochs=5, batch_size=256, learning_rate=0.01,neg_per_pos_sample=1.0):
        #sub models: [similarity, GMF, MLP]
        # import tensorflow as tf
        # tf.set_random_seed(1)
        #super().__init__()
        self.predictions = None
        self.m = m.copy()
        self.name=name
        self.predictions_pickle_file_name=None
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.nn_predicted_mat=None

        self.dropout = dropout
        self.epochs=epochs
        self.mul_emb_size = mul_emb_size
        self.test_tuples = test_tuples
        self.color = 'black'
        self.linestyle = '-'
        self.linewidth = 1
        self.validation_features=validation_tuples
        self.validation_target_att=validation_target
        m_train_array = np.squeeze(np.asarray(self.m))
        self.train_vector_list = dict([(x, m_train_array[x, :]) for x in range(self.m.shape[0])])
        self.propgation_factor=propagation_factor
        #features_creator = graph_features_creator(m)
        #self.features_dict = features_creator.get_normalized_feature_dict()
        #self.num_features = len(self.features_dict[(0, 0)])
        self.num_of_drugs = self.m.shape[0]
        self.neg_per_pos_sample=neg_per_pos_sample

        #features_file_path = r'pickles\filename.pickle'#TODO: create this file using code
        #with open(features_file_path, 'rb') as handle:
        #self.features_dict = pickle.load(handle)
        #self.num_feautres = len(next(iter(self.features_dict.items()))[1])

    def get_sample_train_validation(self, train_pos, train_neg, validation_pos, validation_neg, neg_to_pos_ratio=1.0):
        if neg_to_pos_ratio is None:
            train, validation = train_pos + train_neg, validation_pos + validation_neg
        else:
            train = list()
            # validation = list()
            # random.shuffle(validation_pos);random.shuffle(validation_neg)
            # random.shuffle(train_pos);random.shuffle(train_neg)
            #         train = sample_each_drug_once(train_pos,train)
            #         train = sample_each_drug_once(train_neg,train)
            #         validation = sample_each_drug_once(validation_pos,validation)
            #         validation = sample_each_drug_once(validation_neg,validation)
            train = list(train_pos)
            # validation = list(validation_pos)
            if len(train_pos) * neg_to_pos_ratio < len(train_neg):
                train += random.sample(train_neg, int(len(train_pos) * neg_to_pos_ratio))
            else:
                print('not sampling due to increased number of positive samples')
                train += train_neg
                # validation += random.sample(validation_neg, len(validation_pos))
            validation = validation_pos + validation_neg

        train = [(x[0], x[1]) if random.random() > 0.5 else (x[1], x[0]) for x in
                 train]  # this is redundent now as shared layer is used
        validation = [(x[0], x[1]) if random.random() > 0.5 else (x[1], x[0]) for x in validation]
        return train, validation

    def create_pos_neg_instances(self,train_tuples, validation_tuples, m_train,m_validation):
        train_pos = [x for x in train_tuples if m_train[x[0], x[1]] == 1]
        train_neg = [x for x in train_tuples if m_train[x[0], x[1]] == 0]
        validation_pos = [x for x in validation_tuples if m_validation [x[0], x[1]] == 1]
        validation_neg = [x for x in validation_tuples if m_validation [x[0], x[1]] == 0]
        print(
            f'train pos: {len(train_pos)}, train neg: {len(train_neg)}, val pos: {len(validation_pos)}, val neg: {len(validation_neg)}')
        return train_pos, train_neg, validation_pos, validation_neg

    def fit(self):
        self.init_nn_model()
        self.fit_nn_model()


    def get_embeddings(self):
        return [numpy.concatenate([a, b]) for a,b in zip(self.mult_dense.get_weights()[0],self.mlp.get_weights()[0])]

    def get_instances(self, tuples_sample, m):
        instance_features = []
        instance_features.append(np.array([t[0] for t in tuples_sample]))
        instance_features.append(np.array([t[1] for t in tuples_sample]))
        target_att = np.array([[m[t[0], t[1]]] for t in tuples_sample])
        return instance_features, target_att

    def init_nn_model(self):
        input_node_a = Input(shape=(1,), name='b')
        input_node_b = Input(shape=(1,), name='c')

        regularization = 0
        mlp_emb = Embedding(output_dim=1, name='MLP_embedding',input_dim=self.num_of_drugs,embeddings_regularizer=l2(regularization))
        self.mlp = mlp_emb
        emb_mlp1 = mlp_emb(input_node_a)
        emb_mlp2 = mlp_emb(input_node_b)
        l_mlp = Add()([emb_mlp1,emb_mlp2])

        mult_dense = Embedding(output_dim=self.mul_emb_size, name='GMF_embedding',embeddings_regularizer=l2(regularization),input_dim=self.num_of_drugs) #
        self.mult_dense = mult_dense
        emb_mult1 = mult_dense(input_node_a)
        emb_mult2 = mult_dense(input_node_b)
        dr_emb_mult1 = Dropout(self.dropout)(emb_mult1)
        dr_emb_mult2 = Dropout(self.dropout)(emb_mult2)
        mult = Dropout(0)(Multiply()([dr_emb_mult1, dr_emb_mult2]))
        final_layers = Concatenate(axis=-1)([x for i, x in enumerate([Flatten()(l_mlp), Flatten()(mult)])])
        main_output= Dense(1, activation='sigmoid')(final_layers)#the init is critical for the model to work
        model_emb = Model(inputs=[input_node_a,input_node_b], outputs=main_output)  # fixed_input
        model_emb.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['mae'])  # binary_crossentropy

        #from keras.utils.vis_utils import plot_model
        #import os
        #os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        #plot_model(model_emb, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # self.model_gmf = Model(inputs=[input_node_a,input_node_b], outputs=mult_output) #fixed_input
        # self.model_gmf.compile(optimizer=Adam(),loss='binary_crossentropy', metrics=['accuracy','mae']) #binary_crossentropy
        #
        # self.model_mlp = Model(inputs=[input_node_a,input_node_b], outputs=n_con_output) #fixed_input
        # self.model_mlp.compile(optimizer=Adam(),loss='binary_crossentropy', metrics=['accuracy','mae']) #binary_crossentropy

        self.model_emb = model_emb

    def fit_nn_model(self):

        learning_rate = self.learning_rate
        batch_size = self.batch_size
        epochs=self.epochs
        train_tuples, validation_tuples,self.m_train = create_train_validation_split(self.m, train_ratio=1)  # if ratio = 1 then no validation
        #train_tuples, validation_tuples, self.m_train = create_train_validation_split_single_sample_per_drug(self.m,train_ratio=0.75)


        train_pos, train_neg, validation_pos, validation_neg = self.create_pos_neg_instances(train_tuples, validation_tuples,self.m_train, self.m)

        cnt_epoch=0
        current_learning_rate = learning_rate
        while epochs > cnt_epoch:
            cnt_epoch+=1
            print(f"Epoch number {cnt_epoch} with LR {current_learning_rate}")
            K.set_value(self.model_emb.optimizer.lr, learning_rate)
            # create sample instances#
            train_tuples_sample, validation_tuples_sample = self.get_sample_train_validation(train_pos, train_neg,
                                                                                             validation_pos, validation_neg,
                                                                                             neg_to_pos_ratio=self.neg_per_pos_sample)
            train_features, train_target_att = self.get_instances(train_tuples_sample,self.m_train)
            if self.validation_features==None:
                self.validation_features, self.validation_target_att = self.get_instances(validation_tuples_sample,self.m,)

            if self.validation_features!=None and len(self.validation_target_att)>0:
                self.model_emb.fit(x=train_features, y=train_target_att, batch_size=batch_size, epochs=1,verbose=2,validation_data=(self.validation_features, self.validation_target_att) ) # ,callbacks=[earlycurrent_learning_rateop]
                y_pred = self.model_emb.predict(self.validation_features,batch_size=50000)
                auc = roc_auc_score(self.validation_target_att, y_pred)
                print(f'auc: {auc}')
            else:
                self.model_emb.fit(x=train_features, y=train_target_att, batch_size=batch_size, epochs=1,
                                  verbose=2)  # ,validation_data=(self.validation_features, self.validation_target_att))  # ,callbacks=[earlycurrent_learning_rateop]

        self.w = self.mult_dense.get_weights()
        if sum(self.validation_target_att)>0:
            y_pred = self.model_emb.predict(self.validation_features)
            auc = roc_auc_score(self.validation_target_att, y_pred)
            best_auc = None
            best_x = None
            print(
                f"new evalm before: {auc}, {self.model_emb.evaluate(self.validation_features,self.validation_target_att,verbose=0)}")
            results = []
            for i in range(1,1+1):
                for x in [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
                    self.update_weights(orig_ratio=x,iterations=i)
                    y_pred = self.model_emb.predict(self.validation_features,batch_size=50000)
                    auc = roc_auc_score(self.validation_target_att, y_pred)
                    loss = self.model_emb.evaluate(self.validation_features, self.validation_target_att, verbose=0,batch_size=50000)[0]
                    print(f"new evalm {x}, {i}: {auc}, {loss}")
                    results.append((x,i,auc,loss))
                    if best_auc == None or auc>best_auc:
                        best_auc=auc
                        best_x=x
            print(f'best propagation AUC {best_auc}, best factpr: {best_x}')
            print("results=",results)
        if self.propgation_factor!=None:
            print(f'setting weights to {self.propgation_factor}')
            self.update_weights(orig_ratio=self.propgation_factor)
        print("DONE!")

    def update_weights(self,orig_ratio,iterations=1):
        self.mult_dense.set_weights(self.w)

        for x in range(iterations):
            #y_pred = self.model_emb.predict(self.validation_features, batch_size=100000)
            #first_loss = self.model_emb.evaluate(self.validation_features,self.validation_target_att, batch_size=100000,verbose=0)[0]
            #first_loss = roc_auc_score(self.validation_target_att, y_pred)
            w = self.mult_dense.get_weights()
            new_w =[]
            replaced_count = 0
            for v1 in range(self.num_of_drugs):

                new_node_emb = np.array(w[0][v1])
                new_w.append(new_node_emb)

            G = nx.from_numpy_matrix(self.m_train)
            for v1 in range(self.num_of_drugs):
                if len(G[v1])>0:
                    new_node_emb = np.array(w[0][v1])
                    other_nodes_w = np.zeros(len(new_node_emb)) #empty vector
                    total_weights=0
                    #print(other_nodes_w)
                    #w2 = 0
                    for v2 in G[v1]:
                        curr_weight = 1/len(G[v1])#
                        #w2+=len(G[v2])
                        total_weights+=curr_weight
                        other_nodes_w += curr_weight *w[0][v2]
                    #w1 = len(G[v1])
                    #w2 /= len(G[v1])
                    new_node_emb = new_node_emb*orig_ratio + (1-orig_ratio)*other_nodes_w/total_weights #   here the orig_ratio is 1-alpha from the paper.
                    #new_w.append(new_node_emb*orig_ratio + (1-orig_ratio)*other_nodes_w/total_weights)
                else:
                    new_node_emb = np.array(w[0][v1])
                    #new_w.append(np.array(w[0][v1]))
                #old_w = new_w[v1]
                #new_w[v1] = new_node_emb
                #self.mult_dense.set_weights(np.array([np.array(new_w)]))
                new_w[v1] = new_node_emb
            #     w[0][v1] = new_node_emb
            #     self.mult_dense.set_weights(w)
            #     #y_pred = self.model_emb.predict(self.validation_features,batch_size=100000)
            #     loss = self.model_emb.evaluate(self.validation_features,self.validation_target_att, batch_size=100000,verbose=0)[0]
            #     #auc = roc_auc_score(self.validation_target_att, y_pred)
            #     print(f'Loss for {v1}, {loss }')
            #
            #     w[0][v1] = new_w[v1] #new_w now have the original weight
            #     self.mult_dense.set_weights(w)
            #     if loss < first_loss:
            #         new_w[v1] = new_node_emb
            #         replaced_count+=1
            # print(f'replaced embedding: {replaced_count}')
            self.mult_dense.set_weights(np.array([np.array(new_w)]))

        # for x in range(iterations):
        #     w = self.mlp.get_weights()
        #     new_w =[]
        #     G = nx.from_numpy_matrix(self.m_train)
        #     for v1 in range(self.num_of_drugs):
        #         if len(G[v1])>0:
        #             new_node_emb = np.array(w[0][v1]) * orig_ratio
        #             for v2 in G[v1]:
        #                 new_node_emb += (1-orig_ratio)*(1/len(G[v1]))*w[0][v2]
        #             new_w.append(new_node_emb)
        #         else:
        #             new_w.append(np.array(w[0][v1]))
        #     self.mlp.set_weights(np.array([np.array(new_w)]))

            # for x in range(iterations):
            #     w = self.mult_bias.get_weights()
            #     new_w = []
            #     G = nx.from_numpy_matrix(self.m_train)
            #     for v1 in range(self.num_of_drugs):
            #         if len(G[v1]) > 0:
            #             new_node_emb = np.array(w[0][v1]) * orig_ratio
            #             for v2 in G[v1]:
            #                 new_node_emb += (1 - orig_ratio) * (1 / len(G[v1])) * w[0][v2]
            #             new_w.append(new_node_emb)
            #         else:
            #             new_w.append(np.array(w[0][v1]))
            #     self.mult_bias.set_weights(np.array([np.array(new_w)]))

            #print('bias values:',str(self.mult_bias.get_weights()))


    def predict(self):

        # import math
        # import train_generator
        # importlib.reload(train_generator)
        # from train_generator import d2d_generators

        # d2d_generators_object = d2d_generators(m_train,train_tuples,validation_tuples)

        batch_size = 100000
        # m_evaluation = m_train
        # evaluation_tuples = validation_tuples
        m_evaluation = self.m
        #evaluation_tuples = self.test_tuples  # test_tuples
        evaluation_tuples = [(x,y) for y in range(self.num_of_drugs) for x in range(y+1,self.num_of_drugs)]#self.test_tuples  # test_tuples
        print(f'evaluating {len(evaluation_tuples)} instances')
        eval_instances, _ = self.get_instances(evaluation_tuples,self.m)
        # preds_him = get_pred_him()
        print('done creating instances')
        preds = self.model_emb.predict(eval_instances,
                                  batch_size=batch_size)  # [preds_him[x[0],x[1]] for x in evaluation_tuples] #
        print('done predicting', len(preds))
        count = 0
        predictions = np.zeros((self.m.shape[0], self.m.shape[1]))
        for i in range(len(evaluation_tuples)):
            idx1 = evaluation_tuples[i][0]
            idx2 = evaluation_tuples[i][1]
            predictions[idx1, idx2] = preds[i][0]
            count += 1
        eval_instances, _ = self.get_instances([(x[1], x[0]) for x in evaluation_tuples],self.m)
        preds = self.model_emb.predict(eval_instances,
                                  batch_size=batch_size)  # [preds_him[x[1],x[0]] for x in evaluation_tuples]#
        for i in range(len(evaluation_tuples)):
            idx1 = evaluation_tuples[i][0]
            idx2 = evaluation_tuples[i][1]
            predictions[idx2, idx1] = preds[i][0]  # must be reversed...
            count += 1
        for i in range(self.m.shape[0]):
            for j in range(i + 1, self.m.shape[0]):
                from scipy import stats
                # new_score = stats.hmean([ max(0.000001,predictions[i,j]),max(0.000001,predictions[j,i]) ])
                new_score = (predictions[i, j] + predictions[j, i]) / 2
                # new_score = max(predictions[i,j],predictions[j,i])
                # new_score = predictions[j,i]
                predictions[i, j] = new_score
                predictions[j, i] = new_score
        for i in range(self.m.shape[0]):
            assert predictions[i, i] == 0



        # predictions2 = np.zeros((self.m.shape[0], self.m.shape[1]))
        # G = nx.from_numpy_matrix(self.m)
        # for v1,v2 in  [(i, j) for i in G.nodes() for j in G.nodes() if j>i]:
        #     nn_preds1 = []
        #     for z in G[v2]:
        #         nn_preds1.append(predictions[v1, z])
        #     nn_preds2 = []
        #     for z in G[v1]:
        #         nn_preds2.append(predictions[v2, z])
        #     predictions2[v1, v2] = 0.5*predictions[v1, v2] + 0.5*np.mean(np.nan_to_num([np.mean(nn_preds1),np.mean(nn_preds2)]))
        #     predictions2[v2, v1] = predictions2[v1, v2]

        #predictions = predictions2

        self.predicted_mat = predictions
        print('predicted: ', count)
        s = set(self.test_tuples)
        predictions = [(i, v) for (v, i) in sorted([(v, i) for (i, v) in np.ndenumerate(predictions)],
                                                   reverse=True)]  # get the cells of matrix in ascending order of cell value
        print(1)
        predictions = [(t, v) for t, v in predictions
                       if (t[0], t[1]) in s ]  # just half of the matrix and predictions larger than 0
        if self.predictions_pickle_file_name!=None:
            nn_predicted_mat = self.predicted_mat
            pickling_on = open(self.predictions_pickle_file_name, "wb")
            pickle.dump(nn_predicted_mat, pickling_on)
            pickling_on.close()

        return predictions



class drugs_tanimoto_predictor():
    #implementation of doi:10.1371/journal.pone.0058321
    def __init__(self,m,test_tuples):
        #super().__init__(m, )
        name='Vilar et al.'
        self.predictions = None
        self.m = m.copy()
        self.name=name

        self.test_tuples = test_tuples
        self.color = 'black'
        self.linestyle = ':'
        self.linewidth = 1.5
        self.predictions_pickle_file_name=None
        self.predicted_mat=None

    @staticmethod
    def convert_dist_to_similarity(m2):
        def f(x): return 1 - x;
        f = np.vectorize(f)
        m2 = f(m2)
        return m2

    def fit(self): #100 is best for random spit on latest data. 50 is best for release split on 5.0.2 vs 5.0.11
        print('calculating pairwise distance matrix')
        m1= np.mat(self.m,dtype=bool)
        print('nan m1:', np.sum(np.isnan(m1)))
        m2 = np.mat(pairwise_distances(m1, metric='jaccard'),dtype=float)
        m2 = self.convert_dist_to_similarity(m2)
        np.fill_diagonal(m2,0)
        print('nan m2:', np.sum(np.isnan(m2)))
        m3 = m1 * m2 #mat mul
        np.fill_diagonal(m3,0)
        make_mat_sym(m3,max)
        print('nan m3:',np.sum(np.isnan(m3)))
        self.m3 = np.nan_to_num(m3)



    def predict(self):
        self.predictions = self.m3
        self.predicted_mat = self.m3
        s = set(self.test_tuples)
        predictions = [(i, v) for (v, i) in sorted([(v, i) for (i, v) in np.ndenumerate(self.m3)],
                                                   reverse=True)]  # get the cells of matrix in ascending order of cell value
        print(1)
        predictions = [(t, v) for t, v in predictions
                       if (t[0], t[1]) in s ]  # just half of the matrix and predictions larger than 0
        if self.predictions_pickle_file_name!=None:
            nn_predicted_mat = self.predictions
            pickling_on = open(self.predictions_pickle_file_name, "wb")
            pickle.dump(nn_predicted_mat, pickling_on)
            pickling_on.close()
        return predictions

        # """predicts links in descinding order. only half of the matrix"""
        # print('predicting')
        # predictions = [(i,v) for (v, i) in sorted([(v, i) for (i, v) in np.ndenumerate(self.m3)] , reverse=True)] #get the cells of matrix in ascending order of cell value
        # predictions = [(t,v) for t,v in predictions if t[0]>t[1] and self.m[t[1],t[0]] == 0 ] # just half of the matrix
        # print('done predicting')
        # return predictions