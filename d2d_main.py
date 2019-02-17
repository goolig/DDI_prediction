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
#seed = 8
seed = 123456
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(seed)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
########################################


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

experiment_type = 'retrospective' # 'prediting_final_version' 'holdout' 'retrospective' 'ddi_structural_compare' 'retrospecrive_validation'

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
    models = ['AMF','AMFP', 'Vilar','xgb','features']
elif experiment_type == 'ddi_structural_compare':
    evaluation_method ='Retrospective'
    new_version="5.1.1"
    old_version = new_version
    models = ['AMFP']
else:
    raise Exception('the experiment was not found')


#spliting to train\test
if evaluation_method == 'Retrospective':
    m_test,m_train,evaluator,test_tuples, i2d,evaluation_type,drug_id_to_name  = create_train_test_split_relese(old_relese = old_version,new_relese=new_version)
else:
    m_test,m_train,evaluator,test_tuples, i2d, evaluation_type,drug_id_to_name = create_train_test_split_ratio(new_version,train_ratio,validation_ratio,test_ratio)

print(f'test size: {len(test_tuples)}')
number_of_drugs = len(i2d)

if evaluation_method == 'Retrospective':
    nn_p_file = os.path.join('results','predictions',evaluation_method + old_version + new_version + "nn_predictions.p" )
    tanimoto_p_file = os.path.join('results','predictions',evaluation_method + old_version + new_version + "tanimoto_predictions.p")
else:
    nn_p_file = os.path.join('results','predictions' +evaluation_method + new_version + "nn_predictions.p")
    tanimoto_p_file = os.path.join('results','predictions' + evaluation_method + new_version + "tanimoto_predictions.p")


if evaluation_method == 'Retrospective':
    amfp_params = {'mul_emb_size' : 256, 'dropout':0.3, 'epochs':5, 'batch_size':512, 'learning_rate':0.01,'propagation_factor':0.3}
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
    if old_version=='4.1' and new_version == '5.0.0': #for validation i use the answers to print the results using varying propagation factor
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
    if old_version=='4.1' and new_version == '5.0.0': #for validation i use the answers to print the results using varying propagation factor
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
    for f_name in ['Average common neighbors', 'Average Jaccard coefficient','Adamic Adar','Katz']:
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