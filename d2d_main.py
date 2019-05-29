import matplotlib.pyplot as plt
import random
import numpy
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import numpy as np
from d2d_evaluation import create_train_test_split_relese, create_train_test_split_ratio, average_precision_at_k, write_AUC_output, image_ext, dpi
from d2d_graph_features_creator import graph_features_creator
from d2d_predict import drugs_nn_predictor, drugs_tanimoto_predictor, drugs_xgboost_predictor, drugs_single_feature_predictor
import pandas as pd


# allow reproducing the results:
import os
from keras import backend as K
import tensorflow as tf
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(seed)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
########################################
remove_random=False #remove randomly interaction to miic the CYP interaction removed.
remove_CYP=False #removed shared CYP enzemes interactions
outputs = []
overall_precision = []
overall_k_precision = []
overall_auc = []
overall_ap = []
overall_aupr = []
overall_fpr_tpr = []
k_for_per_drug_precision = 5
k_for_overall_precision=100
predictors = []

def write_embeddings_to_file():
    global pred
    for pred in predictors:
        if pred.name == 'AMF' or pred.name == 'AMFP':
            embeddings = [numpy.concatenate([[i2d[i]], x]) for i, x in enumerate(pred.get_embeddings())]
            path = os.path.join('results','embeddings','embeddings_'+ pred.name + '_' + evaluation_method + '_' +new_version + '.csv')
            pd.DataFrame(embeddings).to_csv(path, index=False)
            print(f'writing embeddings for {pred.name}')


def add_predictor_evaluation(preds,name):
    precision, recall, class_correct = evaluator.get_precision_recall(m_test, preds, test_tuples)
    class_correct_natural_sort = [x for _, x in sorted(zip(preds, [int(x * 1) for x in class_correct]))]
    preds_natural_sort = sorted(preds)
    outputs.append((class_correct_natural_sort, [x[1] for x in preds_natural_sort], name))

    precision_per_drug = evaluator.get_precision_per_drug(m_test, preds, k=k_for_per_drug_precision)
    overall_precision.append((precision, name))
    overall_k_precision.append((precision_per_drug, name))
    # AUC
    pr = [x[1] for x in preds]
    fpr, tpr, _ = roc_curve(class_correct, pr)
    auc = roc_auc_score(class_correct, pr)
    overall_auc.append((auc, name))
    average_precision = average_precision_at_k(k_for_overall_precision, class_correct)
    overall_ap.append((average_precision, name))
    overall_aupr.append(average_precision_score(class_correct, pr))
    overall_fpr_tpr.append((fpr, tpr, name))

def create_plots():
    try:
        font = {'size': 8}
        plt.rc('font', **font)
        # style = [(x.linestyle,x.color,x.linewidth) for x in predictors]
        fig = plt.figure(figsize=(3.5, 8.75), dpi=dpi)
        fig.set_tight_layout(True)
        ax = plt.subplot2grid((3, 1), (0, 0), colspan=1, rowspan=1, fig=fig)
        ax.text(-0.1, 1.05, 'A', transform=ax.transAxes, size=10, weight='bold')
        evaluator.print_roc(overall_fpr_tpr, ax)
        ax = plt.subplot2grid((3, 1), (1, 0), colspan=1, rowspan=1, fig=fig)
        ax.text(-0.1, 1.05, 'B', transform=ax.transAxes, size=10, weight='bold')
        evaluator.print_evaluation(overall_k_precision, ax, max_k=k_for_per_drug_precision,
                                   title='Average precision @ n')  # legend_loc=1
        ax = plt.subplot2grid((3, 1), (2, 0), colspan=1, rowspan=1, fig=fig)
        ax.text(-0.1, 1.05, 'C', transform=ax.transAxes, size=10, weight='bold')
        evaluator.print_evaluation(overall_precision, ax, max_k=k_for_overall_precision, title="Precision @ n")
        fig.savefig( os.path.join('results','figures' + '.' + image_ext),format=image_ext,dpi=dpi,bbox_inches="tight")

    except:
        print('problem creating plots')
def export_predictions(mat,i2d):
    try:
        df = pd.DataFrame(mat, index=i2d, columns=i2d)
        df.to_csv(os.path.join('pickles','predictions.csv'))
    except:
        print('error during export')

def remove_CYP_interactions(i2d, test_tuples, drug_id_to_genname,m_test):
    cyp_enzs = {'CYP1A2', 'CYP2B6', 'CYP2C8', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP2E1', 'CYP3A4', 'CYP3A5', 'CYP3A7'}
    removed=0
    removed_pos=0
    ans = set()
    num_interactions = len(test_tuples)
    for index1,d1 in enumerate(i2d):
        for index2, d2 in enumerate(i2d):
            if index1>index2:
                shared_genes = drug_id_to_genname[d1] & cyp_enzs & drug_id_to_genname[d2]
                ans = ans | shared_genes
                t=(index1,index2)
                if len(shared_genes)>0 and t in test_tuples:
                    #if not remove_random_mimic_cyp:
                    test_tuples.remove(t)
                    removed+=1
                    if m_test[index1,index2]==1:
                        removed_pos+=1
    print('removed with shared enzymes:', removed)
    print('existing interaction removed:',removed_pos)
    removed_ratio = removed/num_interactions
    print('removed ratio =',removed_ratio  )
    # removed=0
    # if remove_random:
    #     for index1, d1 in enumerate(i2d):
    #         for index2, d2 in enumerate(i2d):
    #             if index1<index2 and m_train[index1, index2] == 1 and m_train[index2, index1] == 1 and random.random()<removed_ratio :
    #                 m_train[index1, index2] = 0
    #                 m_train[index2, index1] = 0
    #                 removed += 1
    #     print(remove_random, 'removed with shared enzymes:', removed)
    print(ans)

experiment_type = 'retrospective_cyp' # 'prediting_final_version' 'holdout' 'retrospective' 'ddi_structural_compare' 'retrospecrive_validation' 'retrospective_cyp' 'retrospective_remove_random'

if experiment_type == 'prediting_final_version':
    evaluation_method ='Retrospective'
    new_version="5.1.1"
    old_version = new_version
    models = ['AMFP']
elif experiment_type == 'holdout':
    evaluation_method ='Holdout'
    new_version="5.1.1"
    old_version = "5.0.0"
    train_ratio = 0.7
    validation_ratio = 0.0
    test_ratio = 0.3
    models = ['AMF', 'Vilar','xgb','features']
elif experiment_type == 'retrospective':
    evaluation_method ='Retrospective'
    new_version="5.1.1"
    old_version = "5.0.0"
    models = ['AMF','AMFP', 'Vilar','xgb','features']
elif experiment_type == 'retrospecrive_validation':
    evaluation_method ='Retrospective'
    new_version='5.0.0'
    old_version = "4.1"
    models = ['AMFP']
elif experiment_type == 'ddi_structural_compare':
    evaluation_method ='Retrospective'
    new_version="5.1.1"
    old_version = new_version
    models = ['AMFP']
elif experiment_type == 'retrospective_cyp':
    evaluation_method ='Retrospective (no CYP)'
    new_version="5.1.1"
    old_version = "5.0.0"
    models = ['AMFP','AMF', 'Vilar']
    remove_CYP=True
elif experiment_type == 'retrospective_remove_random':
    evaluation_method = 'Retrospective (remove random)'
    new_version = "5.1.1"
    old_version = "5.0.0"
    models = ['AMFP', 'AMF', 'Vilar']
    remove_random=True
else:
    raise Exception('the experiment was not found')


#spliting to train\test
if evaluation_method == 'Retrospective' or evaluation_method =='Retrospective (no CYP)' or evaluation_method=='Retrospective (remove random)':
    m_test,m_train,evaluator,test_tuples, i2d,evaluation_type,drug_id_to_name,drug_id_to_genname_new = create_train_test_split_relese(old_relese = old_version,new_relese=new_version)
else:
    m_test,m_train,evaluator,test_tuples, i2d, evaluation_type,drug_id_to_name = create_train_test_split_ratio(new_version,train_ratio,validation_ratio,test_ratio)

test_tuples=set(test_tuples)
if remove_CYP:
    remove_CYP_interactions(i2d,test_tuples,drug_id_to_genname_new,m_test)


print(f'test size: {len(test_tuples)}')
number_of_drugs = len(i2d)

if evaluation_method == 'Retrospective':
    nn_p_file = os.path.join('results','predictions',evaluation_method + old_version + new_version + "nn_predictions.p" )
    tanimoto_p_file = os.path.join('results','predictions',evaluation_method + old_version + new_version + "tanimoto_predictions.p")
else:
    nn_p_file = os.path.join('results','predictions' +evaluation_method + new_version + "nn_predictions.p")
    tanimoto_p_file = os.path.join('results','predictions' + evaluation_method + new_version + "tanimoto_predictions.p")


if evaluation_method == 'Retrospective' or evaluation_method =='Retrospective (no CYP)' or evaluation_method=='Retrospective (remove random)':
    amfp_params = {'mul_emb_size' : 512, 'dropout':0.4, 'epochs':6, 'batch_size':1024, 'learning_rate':0.01,'propagation_factor':0.4}
    amf_params = {'mul_emb_size' : 64, 'dropout':0.5, 'epochs':5, 'batch_size':512, 'learning_rate':0.01,'propagation_factor':None}
    xgboost_params = {'colsample_bytree':0.3, 'n_estimators':60, 'subsample':0.8, 'learning_rate':0.01, 'max_depth':5}

else:
    amfp_params = {}
    amf_params = {'mul_emb_size': 256, 'dropout': 0.3, 'epochs': 6, 'batch_size': 256, 'learning_rate': 0.01,'propagation_factor':None}
    xgboost_params = {'colsample_bytree':0.5, 'n_estimators':100, 'learning_rate':0.05}




#AMF
if 'AMF' in models:
    validaition_instance_features=None
    target_att=None
    if experiment_type == 'retrospecrive_validation': #for validation i use the answers to print the results using varying propagation factor
        validaition_instance_features = []
        validaition_instance_features.append(np.array([t[0] for t in test_tuples]))
        validaition_instance_features.append(np.array([t[1] for t in test_tuples]))
        target_att = np.array([[m_test[t]] for t in test_tuples])
    nn = drugs_nn_predictor(m_train,test_tuples,validation_tuples=validaition_instance_features,validation_target=target_att,name='AMF', **amf_params)
    nn.predictions_pickle_file_name = nn_p_file
    predictors.append(nn)

#AMFP
if 'AMFP' in models:
    validaition_instance_features=None
    target_att=None
    if experiment_type == 'retrospecrive_validation': #for validation i use the answers to print the results using varying propagation factor
        validaition_instance_features = []
        validaition_instance_features.append(np.array([t[0] for t in test_tuples]))
        validaition_instance_features.append(np.array([t[1] for t in test_tuples]))
        target_att = np.array([[m_test[t]] for t in test_tuples])
    nn = drugs_nn_predictor(m_train,test_tuples,validation_tuples=validaition_instance_features,validation_target=target_att, name='AMFP', **amfp_params)
    nn.predictions_pickle_file_name=nn_p_file
    predictors.append(nn)

#Tanimoto
if 'Vilar' in models:
    tanimoto = drugs_tanimoto_predictor(m_train, test_tuples)
    tanimoto.predictions_pickle_file_name = tanimoto_p_file
    predictors.append(tanimoto)

#Ensembel
if 'xgb' in models:
    xgboost = drugs_xgboost_predictor(m_train,test_tuples,**xgboost_params)
    xgboost.nn_pred=nn_p_file
    xgboost.tanimoto_pred=tanimoto_p_file
    predictors.append(xgboost)

if 'features' in models:
    features_creator = graph_features_creator(m_train)
    features_dict = features_creator.get_normalized_feature_dict()
    for f_name in ['Average common neighbors', 'Average Jaccard coefficient','Adamic/Adar','$Katz_b$']:
        predictors.append(drugs_single_feature_predictor(m_train,test_tuples,f_name,features_dict))

for predictor in predictors:
    print(f'working on {predictor.name}')
    predictor.fit()
    predictions = predictor.predict()
    predictions_100 = ', '.join([drug_id_to_name[i2d[t[0]]]+ " and " + drug_id_to_name[i2d[t[1]]] for t, v in predictions[:100]])

    print('done predicting ' + predictor.name)
    print(f'100 predictions: {predictions_100}')
    try:
        add_predictor_evaluation(preds=predictions,name=predictor.name)
    except:
        print('problem adding evaluation')

for i in range(len(overall_auc)):
    print(f'Name: {overall_auc[i][1]}, auc: {overall_auc[i][0]}, map: {overall_ap[i][0]}, aupr: {overall_aupr[i]}, AP@k: {overall_k_precision[i][0][0]},{overall_k_precision[i][0][1]},{overall_k_precision[i][0][2]},{overall_k_precision[i][0][3]},{overall_k_precision[i][0][4]}')

create_plots()
export_predictions(predictors[0].predicted_mat,i2d)
write_AUC_output(outputs,evaluation_method)
write_embeddings_to_file()