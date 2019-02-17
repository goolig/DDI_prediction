import os
import random
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from d2d_preprocess import drugs_preproc
from d2d_releases_reader import d2d_releases_reader
from utils import array_to_dict, print_array_file
import matplotlib.pyplot as plt

image_ext='eps'
dpi = 600


def average_precision_at_k(k, class_correct):
    #return average precision at k.
    #more examples: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    #and: https://www.kaggle.com/c/avito-prohibited-content#evaluation
    #class_correct is a list with the binary correct label ordered by confidence level.
    assert k <= len(class_correct)
    assert k > 0
    score = 0.0
    hits = 0.0
    for i in range(k):
        if class_correct[i]==1:
            hits += 1.0
        score += hits /(i+1.0)
    score /= k
    return score


def create_train_validation_split_rectangle(m_train,train_ratio_x= 0.1,train_ratio_y=0.3):
    train_and_val_tuples = drug_evaluator.get_nnz_tuples_from_marix(m_train, zeros=False)
    train_tuples, validation_tuples = [], []
    print('total train+validatoin tuples:', len(train_and_val_tuples))
    limit_x = int(train_ratio_x * m_train.shape[0])
    limit_y = int(train_ratio_y * m_train.shape[0])
    print(f'num drugs in train x: {limit_x}')
    print(f'num drugs in train y: {limit_y}')

    for t in train_and_val_tuples:
        if t[0]<limit_x and t[1]< limit_y:
            validation_tuples.append(t)
            if m_train[t[0],t[1]]==0:
                train_tuples.append(t) #train also on zeros, as we do in testing
        else:
            train_tuples.append(t)
    return train_tuples,validation_tuples


def create_train_validation_split_single_sample_per_drug(m_train,train_ratio = 0.9):
    train_and_val_tuples = drug_evaluator.get_nnz_tuples_from_marix(m_train, zeros=False)
    random.shuffle(train_and_val_tuples)
    train_tuples, validation_tuples = [], []
    print('total train+validatoin tuples:', len(train_and_val_tuples))
    drugs_used_pos = np.zeros(m_train.shape[0])
    drugs_used_neg = np.zeros(m_train.shape[0])
    if train_ratio ==None:
        count_per_drug = np.array([1]*m_train.shape[0])
    else:
        count_per_drug = np.array(m_train.sum(axis=0).tolist()[0]) * (1-train_ratio)
    m = m_train.copy()
    num_instances_per_drug = np.array(m_train.sum(axis=0).tolist()[0])
    count_train_per_drug = np.array([0] * m_train.shape[0])
    for t in train_and_val_tuples:
        train_tuples.append(t)
        if m_train[t[0],t[1]]==1:
            if num_instances_per_drug[t[0]]!=drugs_used_pos[t[0]]+1 and num_instances_per_drug[t[1]]!=drugs_used_pos[t[1]]+1:
                if drugs_used_pos[t[0]]<count_per_drug[t[0]] or drugs_used_pos[t[1]]<count_per_drug[t[1]]:
                    validation_tuples.append(t)
                    drugs_used_pos[t[0]] = drugs_used_pos[t[0]] + 1
                    drugs_used_pos[t[1]] = drugs_used_pos[t[1]] + 1
                    m[t[0], t[1]] = 0
                    m[t[1], t[0]] = 0
                else:
                    count_train_per_drug[t[0]] += 1
                    count_train_per_drug[t[1]] += 1
            else:
                count_train_per_drug[t[0]] += 1
                count_train_per_drug[t[1]] += 1

        else:
            if drugs_used_pos[t[0]]>0 and drugs_used_pos[t[1]]>0:
                if drugs_used_neg[t[0]]<count_per_drug[t[0]] or drugs_used_neg[t[1]]<count_per_drug[t[1]]:
                    validation_tuples.append(t)
                    drugs_used_neg[t[0]] = drugs_used_neg[t[0]] + 1
                    drugs_used_neg[t[1]] = drugs_used_neg[t[1]] + 1
    print(f'num drugs in pos training:{len(count_train_per_drug.nonzero()[0])}')
    print(f'num drugs in pos validation:{len(drugs_used_pos.nonzero()[0])}' )
    print(f'num drugs in neg validation:{len(drugs_used_neg.nonzero()[0])}')
    print(f'avg drugs used in pos validation:{drugs_used_pos.mean()}')
    print(f'avg drugs used in neg validation:{drugs_used_neg.mean()}')
    return train_tuples,validation_tuples,m



def create_train_validation_split(m_train,train_ratio = 0.9):
    train_and_val_tuples = drug_evaluator.get_nnz_tuples_from_marix(m_train, zeros=False)
    train_tuples, validation_tuples = [], []
    print('total train+validatoin tuples:', len(train_and_val_tuples))
    m = m_train.copy()
    for t in train_and_val_tuples:
        train_tuples.append(t)
        if random.uniform(0, 1) > train_ratio:
            validation_tuples.append(t)
            m[t[0], t[1]] = 0
            m[t[1], t[0]] = 0
    return train_tuples,validation_tuples, m

def validate_intersections(i2d, interactions):
    for d in i2d:
        assert d in interactions
        assert len(interactions[d]) > 0 and d not in interactions[d]

def create_train_test_split_relese(old_relese,new_relese):
    print('reading first file')
    d2d_releases_old = d2d_releases_reader()

    drug_reader_old, drug_preproc_old = d2d_releases_old.read_and_preproc_release(old_relese, force_read_file=False)
    print('num interactions in old version:' ,sum([len(drug_preproc_old.valid_drug_to_interactions[x]) for x in drug_preproc_old.valid_drug_to_interactions])/2)
    print('num drugs old', len(drug_preproc_old.valid_drug_to_interactions))

    validate_intersections(drug_preproc_old.valid_drugs_array, drug_preproc_old.valid_drug_to_interactions)
    print('reading seconds file')
    d2d_releases_new = d2d_releases_reader()
    drug_reader_new, drug_preproc_new = d2d_releases_new.read_and_preproc_release(new_relese, force_read_file=False)
    print('num drugs new', len(drug_preproc_new.valid_drug_to_interactions))
    print('num interactions in new version:' ,sum([len(drug_preproc_new.valid_drug_to_interactions[x]) for x in drug_preproc_new.valid_drug_to_interactions])/2)

    validate_intersections(drug_preproc_new.valid_drugs_array, drug_preproc_new.valid_drug_to_interactions)
    print('preprocessing two versions')
    # interscting_i2d = drug_preproc_old.get_interscting_i2d(drug_preproc_new)
    interactions_older, interactions_newer, interscting_i2d = drug_preproc_old.get_intersecting_intersections(
        drug_preproc_new)
    # interscting_i2d = sorted(list(set(drug_preproc_old.valid_drugs_array) & set(drug_preproc_new.valid_drugs_array)))
    # interactions_older, interactions_newer = drug_preproc_old.valid_drug_to_interactions, drug_preproc_new.valid_drug_to_interactions
    #print('intersecting drugs:', interscting_i2d)
    print('intersecting drugs len: ', len(interscting_i2d))

    validate_intersections(interscting_i2d, interactions_older)
    validate_intersections(interscting_i2d, interactions_newer)
    print('creating train matrix')
    m_train = drugs_preproc.create_d2d_sparse_matrix(interscting_i2d, interactions_older)
    print('creating test matrix')
    m_test = drugs_preproc.create_d2d_sparse_matrix(interscting_i2d, interactions_newer)
    evaluator = drug_evaluator(interscting_i2d, interactions_newer, interactions_older)
    test_tuples = drug_evaluator.get_nnz_tuples_from_marix(m_train, True)
    evaluation_type = 'release'
    assert min(sum(np.asarray(m_train)))>0
    assert min(sum(np.asarray(m_train.T)))>0
    assert min(sum(np.asarray(m_test)))>0
    assert min(sum(np.asarray(m_test.T)))>0



    return m_test, m_train, evaluator, test_tuples, interscting_i2d,evaluation_type,drug_reader_old.drug_id_to_name

def create_train_test_split_ratio(relese,train_ratio=0.7,validation_ratio=0,test_ratio=0.3):
    assert train_ratio+validation_ratio+test_ratio==1
    d2d_releases_r1 = d2d_releases_reader()
    drug_reader1, drug_preproc1 = d2d_releases_r1.read_and_preproc_release(relese, force_read_file=False)
    m_test = drug_preproc1.create_d2d_sparse_matrix(drug_preproc1.valid_drugs_array,drug_preproc1.valid_drug_to_interactions)
    evaluator = drug_evaluator(drug_preproc1.valid_drugs_array, drug_preproc1.valid_drug_to_interactions)
    train_tuples, validation_tuples, test_tuples, i2d = evaluator.create_data_split_tuples(train_ratio,validation_ratio,test_ratio)
    evaluator.print_data_split_summary(m_test, train_tuples, validation_tuples, test_tuples)
    i2d = drug_preproc1.valid_drugs_array
    m_train= evaluator.create_train_matrix(m_test, train_tuples, validation_tuples, test_tuples)
    evaluation_type = 'holdout'
    return m_test, m_train, evaluator, test_tuples, i2d,evaluation_type,drug_reader1.drug_id_to_name


def write_AUC_output(outputs,evaluation_method):
    try:

        for o in outputs:
            with open(os.path.join('results',f'GT_{evaluation_method}_{o[2]}.txt'), 'w') as file_handler:
                for item in o[0]:
                    file_handler.write("{}\n".format(item))
            with open(os.path.join('results', f'pred_{evaluation_method}_{o[2]}.txt'), 'w') as file_handler:
                for item in o[1]:
                    file_handler.write("{}\n".format(item))

        print('done writing auc files')
    except:
        print('error during export')


class drug_evaluator():
    def __init__(self, drugs_array, interactions_newer, interactions_older = None):
        self.drugs_array = drugs_array
        self.interactions_newer = interactions_newer
        self.interactions_older = interactions_older

    @staticmethod
    def get_train_validation_sets(m):
        train_and_val_tuples = drug_evaluator.get_nnz_tuples_from_marix(m, zeros=False)
        validation_tuples,train_tuples = [],[]
        print('total train+validatoin tuples:', len(train_and_val_tuples))
        train_ratio = 0.9
        for t in train_and_val_tuples:
            if random.uniform(0, 1) < train_ratio:
                train_tuples.append(t)
            else:
                validation_tuples.append(t)
        print('train tuples:', len(train_tuples))
        print('validation tuples:', len(validation_tuples))
        return train_tuples,validation_tuples

    @staticmethod
    def get_nnz_tuples_from_marix(m,zeros):

        """returns <x,y> collection of the test tuples. The test are cells containing zero in the original matrix"""
        if zeros:
            res = [i for (i, v) in np.ndenumerate(m) if
                   v == 0 and i[0] > i[1]] #get the cells of matrix in ascending order of cell value
        else:
            res = [i for (i, v) in np.ndenumerate(m) if
                    i[0] > i[1]]  # get the cells of matrix in ascending order of cell value
        return res


    def print_data_split_summary(self,m_full, train_tuples, validation_tuples, test_tuples):
        count_i_train = 0
        count_i_validation = 0
        count_i_test = 0
        for t in train_tuples:
            if m_full[t[0], t[1]] > 0:
                count_i_train += 1
        for t in validation_tuples:
            if m_full[t[0], t[1]] > 0:
                count_i_validation += 1
        for t in test_tuples:
            if m_full[t[0], t[1]] > 0:
                count_i_test += 1
        print('train total: %d, train interactions: %d, train ratio: %f' % (
            len(train_tuples), count_i_train, count_i_train / len(train_tuples) if len(train_tuples)>0 else 0))
        print('validation total: %d, validation interactions: %d, validation ratio: %f' % (
            len(validation_tuples), count_i_validation, count_i_validation / len(validation_tuples) if len(validation_tuples)>0 else 0))
        print('test total: %d, test interactions: %d, test ratio: %f' % (
            len(test_tuples), count_i_test, count_i_test / len(test_tuples) if len(test_tuples)>0 else 0))

    def create_data_split_tuples(self, train_ratio = 0.7, validation_ratio = 0.15,test_ratio = 0.15):
        """generate train,test and validation sets using the number of drugs
        The sets are represented as tuples of the indexes in the matrix <x,y> where x<y"""
        n = len(self.drugs_array)
        print('number of drugs: %d' % n)
        assert train_ratio + validation_ratio + test_ratio == 1

        all_tuples = [(x, y) for x in range(n) for y in range(n) if x < y] #just upper half of the matrix
        train_tuples,validation_tuples ,test_tuples = [],[],[]
        print('number of cells in upper trainagle: %d' % len(all_tuples))

        for t in all_tuples:
            r = random.uniform(0, 1)
            if r < train_ratio:
                train_tuples.append(t)
            elif r < train_ratio + validation_ratio:
                validation_tuples.append(t)
            else:
                test_tuples.append(t)

        print('Only upper half: Train size: %d, validation size: %d, test size %d' % (len(train_tuples), len(validation_tuples), len(test_tuples)))

        # coding the drugs to codes now. this is not neede at the moment, but if we would like to put some
        #more logic into the splitting it will be needed
        assert len(train_tuples)==len(set(train_tuples)) and len(test_tuples)==len(set(test_tuples)) and len(validation_tuples)==len(set(validation_tuples))
        return train_tuples,validation_tuples,test_tuples, self.drugs_array

    def create_train_test_ratio_split(self, train_ratio=0.85,validation_ratio = 0, test_ratio=0.15):
        """
        returns a split of the data using dictionaries.
        The dics are symmetric: y in ans[x] -> x in ans[y]
        a mapping of index to drug name is also given as an array
        """
        assert train_ratio+validation_ratio+test_ratio==0
        train_drug_to_interactions ,test_drug_to_interactions,validation_drug_to_interactions   = {},{},{}
        train_tuples, validation_tuples, test_tuples, i2d = self.create_data_split_tuples(train_ratio,validation_ratio,test_ratio)
        d2i = array_to_dict(i2d)
        train_tuples, validation_tuples, test_tuples = set(train_tuples),set(validation_tuples),set(test_tuples)

        for drug1,drug1_interactions in self.interactions_newer.items():
            for drug2 in drug1_interactions:
                assert drug1 != drug2 #just making sure again
                drug1_index = d2i[drug1]
                drug2_index = d2i[drug2]
                t = (drug1_index,drug2_index)
                if t in train_tuples or tuple(reversed(t)) in train_tuples:
                    train_drug_to_interactions.setdefault(drug1,[]).append(drug2)
                    assert drug2 not in train_drug_to_interactions or drug1 not in train_drug_to_interactions[drug2], 'the drug is already in the list. doing to insert it again'
                    train_drug_to_interactions.setdefault(drug2, []).append(drug1)
                elif t in validation_tuples or tuple(reversed(t)) in validation_tuples:
                    validation_drug_to_interactions.setdefault(drug1,[]).append(drug2)
                    assert drug2 not in validation_drug_to_interactions or drug1 not in validation_drug_to_interactions[drug2], 'the drug is already in the list. doing to insert it again'
                    validation_drug_to_interactions.setdefault(drug2, []).append(drug1)
                else:
                    assert t in test_tuples, 'drug wasnt put anywhere'
                    test_drug_to_interactions.setdefault(drug1,[]).append(drug2)
                    assert drug2 not in test_drug_to_interactions or drug1 not in test_drug_to_interactions[drug2], 'the drug is already in the list. doing to insert it again'
                    test_drug_to_interactions.setdefault(drug2, []).append(drug1)
        return train_drug_to_interactions, validation_drug_to_interactions, test_drug_to_interactions, i2d

    def print_roc(self,overall_fpr_tpr,ax,style=None):
        #fig = plt.figure(figsize=(5.2, 3.9), dpi=600)
        #ax = fig.add_subplot(111)
        for j, ftp_tpr in enumerate(overall_fpr_tpr):
            #ax.plot(ftp_tpr[0], ftp_tpr[1], label=ftp_tpr[2], linestyle=style[j][0],color = style[j][1],linewidth=style[j][2])
            ax.plot(ftp_tpr[0], ftp_tpr[1], label=ftp_tpr[2])
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC")
        ax.legend(loc="upper left", bbox_to_anchor=(1,1))


    def print_evaluation(self, preds,ax,style=None,max_k=100,title="",legend_loc=4):
        #fig = plt.figure(figsize=(5.2, 7.8), dpi=600)
        #ax = fig.add_subplot(111)
        for i,values in enumerate(preds):
            data=values[0][:max_k]
            #print(f"name:{name}, precision@k: {data}")
            #ax.plot(range(1,len(data)+1),data,label=values[1],linestyle=style[i][0],color = style[i][1],linewidth=style[i][2])
            ax.plot(range(1, len(data) + 1), data, label=values[1])
        ax.set_xlabel("n")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        #ax.legend( loc=legend_loc)


    def get_precision_per_drug(self, m_test, predictions,k=5):
        n_drugs = m_test.shape[0]
        per_drug_pred = [[None for i in range(k)] for j in range(n_drugs)]
        cnt_interactions_per_drug = [0 for i in range(n_drugs)]
        for i,t in enumerate(predictions):
            drug_ids = t[0]
            d1 = drug_ids[0]
            d2 = drug_ids[1]
            assert d1!=d2
            if cnt_interactions_per_drug[d1]<k:
                per_drug_pred[d1][cnt_interactions_per_drug[d1]] = d2
                cnt_interactions_per_drug[d1]+=1
            if cnt_interactions_per_drug[d2] < k:
                per_drug_pred[d2][cnt_interactions_per_drug[d2]] = d1
                cnt_interactions_per_drug[d2] += 1

        per_drug_precision= [[None for i in range(k)] for j in range(n_drugs)]
        for d in range(n_drugs):
            tp = 0
            for i in range(1,k+1):
                if per_drug_pred[d][i-1] is not None:
                    if m_test[d,per_drug_pred[d][i-1]] !=0:
                        tp+=1
                    per_drug_precision[d][i-1]=tp/i
                else:
                    assert False, "drugs with no prediction"
        res = np.average(per_drug_precision,axis=0)
        #print(res)
        return res
        #print(cnt_interactions_per_drug)

    def get_precision_recall(self, m_test, predictions, test_tuples):
        N,P=0,0
        for t in test_tuples:
            if m_test[t[0], t[1]] > 0:
                P+=1
            else:
                N+=1
        print('Positives: %d, Negatives: %d' % (P,N))
        print('Positive ratio: %f' % (P/(N+P)))


        precision_at_k, recall_at_k, class_correct  = [],[], []
        t=0
        test_tuples_set = set(test_tuples)
        for i,tuple in enumerate(predictions):
            drug_ids = tuple[0]
            if (drug_ids[0] ,drug_ids[1]) not in test_tuples_set:
                assert False, 'edges were predicted which are from the training set' + str(drug_ids[0]) +" ," + str(drug_ids[1])
            test_tuples_set.remove((drug_ids[0], drug_ids[1]))
            if m_test[drug_ids[0], drug_ids[1]] > 0:
                t+=1
                class_correct.append(True)
            else:
                class_correct.append(False)
            precision_at_k.append(t/(i+1))
            recall_at_k.append(t/P)
        assert len(test_tuples_set)==0, f'unpredicted interactions, {test_tuples_set}'
        print('precision @ cutoff: %f, recall @ cutoff: %f, cutoff: %d' % (precision_at_k[P-1],recall_at_k[P-1],P))
        return precision_at_k,recall_at_k, class_correct
        # predictions = predictions[:P] #the amount we predict is the amount of Trues in the test set TODO: change this
        #
        # TP = []
        # FP = []
        # for drug_ids in predictions:
        #     if m_test[drug_ids[0], drug_ids[1]]>0:
        #         #assert (drug_ids[0],drug_ids[1]) in test_tuples #it is ok, but it takes a very long time
        #         assert m_test[drug_ids[0], drug_ids[1]] != 0
        #         TP.append((drug_ids[0],drug_ids[1]))
        #     else:
        #         assert m_test[drug_ids[1], drug_ids[0]] == 0
        #         FP.append((drug_ids[0],drug_ids[1]))
        # print('tp', len(TP), 'fp', len(FP), 'precision:', len(TP) / (len(TP) + len(FP)))
        # print('count in test',len(test_tuples))
        # assert P == len(TP)+len(FP)
        # #print('TP sample:', TP[:100])
        # #print('FP smaple:', FP[:100])
        # return TP,FP


    def create_train_matrix(self,m_full,train_tuples,validation_tuples,test_tuples):
        m_train = lil_matrix(m_full)
        # for t in validation_tuples:
        #     m_train[t[0],t[1]] = 0
        #     m_train[t[1], t[0]] = 0
        for t in test_tuples:
            m_train[t[0],t[1]] = 0
            m_train[t[1], t[0]] = 0
        print('train matrix non zeros: %d' % m_train.nnz)
        return m_train.todense()



########## dont use

    def create_train_test_ratio_split_deprecared(self, test_ratio=0.1):
        train_drug_array = []
        train_drug_to_interactions={}
        test_drug_array=[]
        test_drug_to_interactions = {}
        count_interaction_test=0
        count_interaction_train=0
        for drug1,drug1_interactions in self.interactions_newer.items():
            for drug2 in drug1_interactions:
                assert drug1 != drug2 #just making sure again
                if (not (drug2 in train_drug_to_interactions and drug1 in train_drug_to_interactions[drug2])) and \
                        (random.uniform(0,1) <test_ratio or
                             (drug2 in test_drug_to_interactions and drug1 in test_drug_to_interactions[drug2])): #TODO: sample only drugs with suffecient interactions

                    test_drug_to_interactions.setdefault(drug1,set()).add(drug2)
                    count_interaction_test+=1
                else:
                    train_drug_to_interactions.setdefault(drug1,set()).add(drug2)
                    count_interaction_train+=1
            assert drug1 in train_drug_to_interactions or drug1 in test_drug_to_interactions
            if drug1 in train_drug_to_interactions:
                train_drug_array.append(drug1)
            if drug1 in test_drug_to_interactions:
                test_drug_array.append(drug1)
        print('train interactions:',count_interaction_train,'test interactions:',count_interaction_test)
        assert count_interaction_train % 2 ==0
        assert count_interaction_test % 2 ==0
        print('test array size:',len(test_drug_array),'train array size:',len(train_drug_array),'test interaction size',len(test_drug_to_interactions),'train interaction size',len(train_drug_to_interactions))
        return (train_drug_array,train_drug_to_interactions),(test_drug_array,test_drug_to_interactions),count_interaction_test



    def evaluate_ratio_using_dics_deprecaed(self, train_data, test_data, predictions):
        count_interaction_test = 0
        for d in test_data:
            count_interaction_test +=len(test_data[d])
        count_interaction_test = count_interaction_test

        predictions = predictions[:count_interaction_test]

        TP = []
        FP = []
        for pred in predictions:
            drug_name1 = train_drug_array[pred[0]]
            drug_name2 = train_drug_array[pred[1]]
            if drug_name1 in test_drug_array and \
                            drug_name2 in test_data[drug_name1]:
                assert drug_name1 in test_data[drug_name2]
                #TP
                TP.append((drug_name1,drug_name2))
            else:
                assert drug_name2 not in test_drug_array or drug_name1 not in test_data[drug_name2]
                FP.append((drug_name1,drug_name2))
        print('tp', len(TP), 'fp', len(FP), 'precision:', len(TP) / (len(FP) + len(FP)))
        print('count interaction test (both rectangles of the matrix)',count_interaction_test)
        print('count interaction test (one rectangles of the matrix)', count_interaction_test/2)
        assert count_interaction_test == len(TP)+len(FP)
        return TP,FP

    def write_pred_to_file(self, class_correct, interscting_i2d, m_train, m_test, predictions):
        nnz_train = np.count_nonzero(np.squeeze(np.asarray(m_train)), axis=1)
        nnz_test = np.count_nonzero(np.squeeze(np.asarray(m_test)), axis=1)
        a = list(zip(list(range(len(class_correct))) + list(range(len(class_correct))),class_correct + class_correct,
                             [interscting_i2d[x[0][0]] for x in predictions] + [interscting_i2d[x[0][1]] for x in predictions],
                            [nnz_test[x[0][0]] for x in predictions] + [nnz_test[x[0][1]] for x in predictions],
                            [nnz_train[x[0][0]] for x in predictions] + [nnz_train[x[0][1]] for x in predictions],

                             # [interscting_i2d[x[0][1]] for x in predictions] + [interscting_i2d[x[0][0]] for x in predictions],
                             # [nnz_test[x[0][1]] for x in predictions] + [nnz_test[x[0][0]] for x in predictions],
                             # [nnz_train[x[0][1]] for x in predictions] + [nnz_train[x[0][0]] for x in predictions],
                             #[len(interactions_older[interscting_i2d[x[0][0]]]) for x in predictions] + [len(interactions_older[interscting_i2d[x[0][1]]]) for x in predictions],

                             #[interscting_i2d[x[0][1]] for x in predictions] + [interscting_i2d[x[0][0]] for x in predictions],
                             #[len(interactions_newer[interscting_i2d[x[0][1]]]) for x in predictions] + [len(interactions_newer[interscting_i2d[x[0][0]]]) for x in predictions],
                             #[len(interactions_older[interscting_i2d[x[0][1]]]) for x in predictions] + [len(interactions_older[interscting_i2d[x[0][0]]]) for x in predictions],
                             [x[1] for x in predictions] + [x[1] for x in predictions]))
        a.insert(0,('IID','True_value',
                    'd_name','d_#interaction_new','d_#interaction_old',
                    #'d2_name','d2_#interaction_new','d2_#interaction_old',
                    'prediction'))
        path = 'predictions.csv'
        print_array_file(a, path)
        return path


