import copy
from scipy.sparse import lil_matrix, coo_matrix, csr_matrix
import numpy as np

from utils import array_to_dict, flatten_list


class drugs_preproc():
    def __init__(self,drug_to_interactions ,all_drugs):
        self.dirty_drug_to_interactions =drug_to_interactions
        self.valid_drug_to_interactions = None
        self.all_drugs = all_drugs
        self.valid_drugs_array=None

    def get_intersecting_intersections(self, new_preproc):

        #assert len(self.valid_drug_to_interactions) > len (other_preproc.valid_drug_to_interactions), 'self should be the newer version. i assume newer versions contains mors drugs'
        intersecting_drugs, drugs_to_remove_older, drugs_to_remove_newer = self.get_intersecting_drugs(new_preproc)

        print('intersecting drugs len: ', len(intersecting_drugs))
        print('removing drugs from older: ', len(drugs_to_remove_older))
        print('removing drugs from newer: ', len(drugs_to_remove_newer))
        print('cleaning older version interactions:')
        interactions_older = drugs_preproc.remove_non_intersecting_interactions(self.valid_drug_to_interactions,drugs_to_remove_older)
        assert len(interactions_older) == len(intersecting_drugs),str(len(interactions_older)) + " "+ str(len(intersecting_drugs))
        print('cleaning newer version interactions:')
        interactions_newer = drugs_preproc.remove_non_intersecting_interactions(new_preproc.valid_drug_to_interactions, drugs_to_remove_newer)
        assert len(interactions_newer) == len(intersecting_drugs),str(len(interactions_newer)) + " "+ str(len(intersecting_drugs))
        drugs_preproc.print_release_difference(interactions_older, interactions_newer)
        return interactions_older, interactions_newer, intersecting_drugs

    @staticmethod
    def print_release_difference(interactions_self, interactions_other):
        added, new_interactions, removed, old_interactions = 0, 0,0,0
        stats_new = []
        for d1,d1_insteractions in interactions_self.items():
            stats_new.append(len(d1_insteractions))
            for d2 in d1_insteractions:
                new_interactions+=1
                if not d2 in interactions_other[d1]:
                    added+=1

        print('count new intercations:',new_interactions, 'added:', added)
        stats_old = []
        for d1,d1_insteractions in interactions_other.items():
            stats_old.append(len(d1_insteractions))
            for d2 in d1_insteractions:
                old_interactions+=1
                if not d2 in interactions_self[d1]:
                    removed+=1

        print('count old intercations:', old_interactions, 'removed:', removed)

    @staticmethod
    def remove_non_intersecting_interactions(interactions, drugs_to_remove):
        interactions_orig = dict(interactions)
        #print('removing',drugs_to_remove)
        for d in drugs_to_remove:
            del interactions_orig[d]
        interactions_self_new = {} #can't change dict while looping, making new one
        original_num_interactions, new_num_interactions = 0,0
        for d1, d1_interactions in interactions_orig.items():
            original_num_interactions+=len(d1_interactions)
            d1_interactions_new = [x for x in d1_interactions if x not in drugs_to_remove]
            new_num_interactions+= len(d1_interactions_new)
            if len(d1_interactions_new)>0:
                interactions_self_new[d1] = d1_interactions_new
        print('original interaction num:', original_num_interactions,'new num:',new_num_interactions,'interactions removed due to new drugs:',original_num_interactions-new_num_interactions)


        return interactions_self_new

    def get_intersecting_drugs(self, newer_preproc):
        intersection = set(self.valid_drugs_array) & set(newer_preproc.valid_drugs_array)
        drugs_to_remove_older =  set(self.valid_drugs_array) - intersection
        drugs_to_remove_newer = set(newer_preproc.valid_drugs_array) - intersection

        interactions_older = drugs_preproc.remove_non_intersecting_interactions(self.valid_drug_to_interactions, drugs_to_remove_older)
        interactions_newer = drugs_preproc.remove_non_intersecting_interactions(newer_preproc.valid_drug_to_interactions, drugs_to_remove_newer)

        intersection = interactions_older.keys() & interactions_newer.keys()
        drugs_to_remove_older = set(self.valid_drugs_array) - intersection
        drugs_to_remove_newer = set(newer_preproc.valid_drugs_array) - intersection

        return sorted(list(intersection)), drugs_to_remove_older, drugs_to_remove_newer

    def calc_valid_drugs_print_summary(self):
        print('analyzing interactions')
        count_valid, count_total,count_not_in_db,count_asymmetric_inter  = 0,0,0,0
        self.valid_drugs_array = []
        not_in_db = set()
        for d1, d1_interactions in self.dirty_drug_to_interactions.items():
            drug_is_valid = False  # d1 is valid if it has at least one interaction with a drug from the db and the interaction is symmetric.
            assert d1 in self.all_drugs
            for d2 in d1_interactions:
                assert d1 != d2
                count_total += 1
                if d2 not in self.all_drugs:
                    count_not_in_db += 1
                    not_in_db.add(d2)
                elif d2 not in self.dirty_drug_to_interactions or d1 not in self.dirty_drug_to_interactions[d2]:
                    count_asymmetric_inter += 1 #TODO: perhaps we do want to force it to be symmetric? it looks like there arent too many anyway...
                else:
                    count_valid += 1
                    drug_is_valid = True
            if drug_is_valid:
                self.valid_drugs_array.append(d1)
        assert count_total == count_valid + count_not_in_db + count_asymmetric_inter
        print('total interactions:', count_total, 'valid interactions:', count_valid, 'count not in db:',
              count_not_in_db, 'count asymmetric interactions', count_asymmetric_inter)
        print('num drugs not in db %d:'%(len(not_in_db)))

    def create_valid_drug_interactions(self):
        assert self.valid_drugs_array is not None # must run get_valid_drugs_print_summary first
        self.valid_drug_to_interactions=copy.deepcopy(self.dirty_drug_to_interactions)
        valid_drug_to_id = array_to_dict(self.valid_drugs_array)
        for d1,d1_interactions in self.dirty_drug_to_interactions.items():
            if d1 not in valid_drug_to_id:
                del self.valid_drug_to_interactions[d1]
            else:
                for d2 in d1_interactions:
                    if d2 not in valid_drug_to_id:
                        self.valid_drug_to_interactions[d1].remove(d2) #remove is not very efficient.
        print('clean interactions are ready')
        assert len(self.valid_drugs_array) == len(self.valid_drug_to_interactions.keys())


    # @staticmethod
    # def create_d2d_sparse_matrix(i2d, drug_to_interactions):
    #     d2i = array_to_dict(i2d)
    #     number_of_drugs = len(d2i)
    #     print('creating matrix')
    #     m = np.matrix(np.zeros(shape=(number_of_drugs,number_of_drugs)),dtype='f')
    #
    #     errors=0
    #     oks=0
    #     for d1 in i2d:
    #         for d2 in drug_to_interactions[d1]:
    #             try:
    #                 id1 = d2i[d1]
    #                 id2 = d2i[d2]
    #                 if id1!=id2:
    #                     m[id1, id2] = 1
    #                     m[id2, id1] = 1
    #                     oks+=1
    #                 else:
    #                     errors+=1
    #             except:
    #                 errors+=1
    #     print(f'oks: {oks}, errors: {errors}')
    #     return m


    @staticmethod
    def create_d2d_sparse_matrix(i2d, drug_to_interactions):
        d2i = array_to_dict(i2d)
        number_of_drugs = len(d2i)
        print('creating matrix')
        rows = flatten_list([[d2i[x[0]]] * len(x[1]) for x in sorted(drug_to_interactions.items())])
        cols = [d2i[t] for t in flatten_list([x[1] for x in sorted(drug_to_interactions.items())])]
        print('number of valid interactions:',len(cols))
        assert len(rows) == len(cols)
        data = [1] * len(cols)
        m = csr_matrix((data,(rows,cols)), shape=(number_of_drugs, number_of_drugs),dtype='f')
        print('m shape:', m.shape, 'm non zeros:', m.nnz)
        m = m.todense()
        count_non_sym=0
        for i in range(m.shape[0]):
            for j in range(i+1,m.shape[0]):
                if m[i,j]!=m[j,i]:
                    count_non_sym+=1
                m[i,j]=max(m[i,j],m[j,i])
                m[j, i] = m[i, j]
        print('non sym count (matrix was made sym using max):',count_non_sym)
        assert np.allclose(m, m.T, atol=1e-8) #matrix is symmetric
        return m
