# Drug-drug interaction detection, AMF & AMFP.

This repository contains code necessary for preprocessing, training and evaluating drug-drug interaction detection systems. In particular AMF and AMFP are implemented here. 
The method was published lately in PLOS One:
Shtar G, Rokach L, Shapira B (2019) Detecting drug-drug interactions using artificial neural networks and classic graph similarity measures. PLOS ONE 14(8): e0219796. https://doi.org/10.1371/journal.pone.0219796

Author: Guy Shtar.

Drug-drug interactions are preventable causes of medical injuries and often result in1doctor and emergency room visits.  Computational techniques can be used to predict2potential drug-drug interactions.  We approach the drug-drug interaction prediction problem as a link prediction problem and present two novel methods for drug-drug interaction prediction based on artificial neural networks and factor propagation over graph nodes:  adjacency matrix factorization (AMF) and adjacency matrix factorization6with propagation (AMFP).

## Data:

The data should be downloaded from [DrugBank](https://www.drugbank.ca/) and placed under `pickles\data\DrugBank_releases\version`. for example, the file containing version 5.0.0 should be placed under: 
`pickles\data\DrugBank_releases\5.0.0\drugbank_all_full_database.xml.zip`

## Running the code:

use Main.py or DDI.ipyn. Tested on Python 3.6.

Results are written to Results folder. In particular figures containing the metrics used in the paper given above and the embeddings created for each drug. The predictions are printed to the screen with the final results. Note: the last entry for each drug embedding contains the Bias value for this drug.

## Files:

* d2d_DAL - Data access layer, reads DrugBank's release and parses it.
* d2d_evaluation - evaluation procedures. splitting the data, calculating final scores, creating graphs.
* d2d_graph_features_creator - creating graph features using the adjacency matrix.
* d2d_main - main program for running expermints.
* d2d_predict - implements all predictors used in the paper. Including AMF, AMFP and the XGBoost model.
* d2d_preprocess - preprocessing DrugBank's data.
* d2d_releases_reader - combining the DAL and preprocess procedures.
* dataset_dates - datasets metadata for all versions of DrugBank used in our research.
* utils - utilities used in the code.
* Zhang DDI Prediction - comparision of AMF, AMFP, XGBoost ensemble to the method propsed by Zhang et al [1]. See readme in folder for more details.
* AMFP_Preds_5.1.4.txt - 1,000 interaction predictions made using DrugBank 5.1.4 and AMFP. Given by confidence level.
* embeddings_AMFP_Retrospective_5.1.4.csv	- Drug embeddings created using DrugBank 5.1.4 and AMFP.

[1] Zhang W, Chen Y, Liu F, Luo F, Tian G, Li X. Predicting potential drug-drug interactions by integrating
chemical, biological, phenotypic and network data. BMC Bioinformatics. 2017;18(1):18.
doi:10.1186/s12859-016-1415-9.

## Install:

`pip3 install -r requirements.txt`
