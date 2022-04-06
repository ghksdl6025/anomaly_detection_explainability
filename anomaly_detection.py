import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score, f1_score
import numpy as np

import matplotlib.pyplot as plt
import os
from sklearn.model_selection import cross_val_score
import json
import pickle as pkl
import utils

df = pd.read_csv('./preprocessed_loan_baseline.pnml_noise_0.09999999999999999_iteration_1_seed_14329_sample.csv')
used_models = 'XGB'


key_pair = {'Case ID':'caseid', 'Activity':'activity', 'Complete Timestamp':'ts'}
df = df.rename(columns=key_pair)

if 'resource' in df.columns.values:
    df = df.loc[:,['caseid','activity','ts','resource','noise']]

else:
    df = df.loc[:,['caseid','activity','ts','noise']]

groups = df.groupby('caseid')
concating = []
max_case_len = max([len(group) for _, group in groups])
caseids = list(set(df['caseid']))

outcome = []
for _, group in groups:
    group = group.reset_index(drop=True)
    actlist = list(group['activity'])
    outcomelist = actlist[1:] + [np.nan]
    group['outcome'] = outcomelist
    concating.append(group)

dfn = pd.concat(concating)

max_case_len =15
idslist = []
for prefix in range(1, max_case_len):
    idslist.append(utils.indexbase_encoding(dfn,prefix))

prefixlist= list(range(1, max_case_len))

acc_dict= {}

print(used_models)
models = []
testdf_list = []


for pos,prefix in enumerate(idslist):  
    caseids = list(set(prefix['caseid']))
    np.random.seed(0)
    trainids = np.random.choice(caseids, int(len(caseids)*0.7), replace=False)
    traindf = prefix[prefix['caseid'].isin(trainids)].reset_index(drop=True)
    testdf = prefix[~prefix['caseid'].isin(trainids)].reset_index(drop=True)
    testdf_list.append(testdf)

    y_train = traindf['outcome']
    x_train = traindf.drop(columns=['outcome','caseid'],axis=1)

    y_test = testdf['outcome']
    x_test = testdf.drop(columns=['outcome','caseid'],axis=1)

    # Random forest result    
    
    if used_models == 'RF':
        m = RandomForestClassifier(n_estimators=10, criterion='entropy').fit(x_train,y_train)
        y_pred = m.predict(x_test)

    elif used_models =='XGB':
        m = xgb.XGBClassifier(n_estimators = 20, learning_rate=0.01).fit(x_train, y_train)
        y_pred = m.predict(x_test)
        
    models.append(m)

    filename = './models/%s prefix %s.pkl'%(used_models, pos+1)
    with open(filename,'wb') as f:
        pkl.dump(m, f)

    acc_dict['prefix_%s'%(str(prefixlist[pos]))] =  accuracy_score(y_test,y_pred)
    
    testids = list(set(testdf['caseid']))
    test_file_name = './data/Prefix %s testdata.pkl'%(str(pos+1))
    with open(test_file_name,'wb') as f:
        pkl.dump(testids,f)

for_confusion_matrix = {}

counting_normal = 0
for threshold in [0.01,0.05,0.1,0.15,0.2,0.25]:
    global_true =[]
    global_pred = []
    ad_predictions=[]
    ad_true = []

    for pos, prefix in enumerate(idslist):
        testing_case_ids = set(testdf_list[-1]['caseid'])

        prediction_list = []
        testing_case_ids = set(testdf_list[pos]['caseid'])
        for caseid in list(testing_case_ids):
            prediction_label = 'Normal'
            x_test = testdf_list[pos][testdf_list[pos]['caseid'] ==caseid]
            true_outcome = x_test['outcome'].values[0]
            
            x_test_features = list(x_test.columns.values)
            x_test_features.remove('caseid')
            x_test_features.remove('outcome')
            
            x_test = x_test.loc[:, x_test_features]
            x_test = np.array(x_test.values).reshape(1,-1)

            model_classes = models[pos].classes_
            predictions_proba = models[pos].predict_proba(x_test)[0]
            predicted_one = model_classes[np.argmax(predictions_proba)]
        
            if predicted_one  == 'Not Available':
                prediction_label = 'Not Available'
            else:
                if true_outcome in model_classes:
                    labelidx = list(model_classes).index(true_outcome)

                    if predictions_proba[labelidx] <threshold:
                        prediction_label = 'Anomalous'
                else:
                    prediction_label = 'Anomalous'
           
            noisedf = df[df['caseid'] == caseid].reset_index(drop=True)
            noiselabel = list(noisedf['noise'])[pos]
            if np.isnan(noiselabel):
                noiselabel= 'Normal'
            else:
                noiselabel= 'Anomalous'
            ad_predictions.append(prediction_label)
            ad_true.append(noiselabel)
        
    for_confusion_matrix[threshold]=[ad_predictions, ad_true]

for t in for_confusion_matrix.keys():
    print(t)
    predictions = for_confusion_matrix[t][0]
    trues = for_confusion_matrix[t][1]
    print(classification_report(y_pred = predictions, y_true = trues))
    print('Accuarcy: ',accuracy_score(y_pred = predictions, y_true = trues))
    print('F1 score: ',f1_score(y_pred = predictions, y_true = trues, average='binary', pos_label='Normal'))
    print(set(predictions), set(trues))