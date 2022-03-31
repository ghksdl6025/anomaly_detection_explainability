#%%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import cross_val_score
import json
import pickle as pkl

#%%
def filter_by_prefix(df,prefix):
    '''
    Filter case by prefix length
    
    Parameters
    ----------
    df : pandas dataframe
        Assigned dataframe to slice by prefix length
    
    prefix : int
        Prefix length to slice to cases in fixed length
    
    Returns
    ----------
    Return dataframe with sliced cases
    '''
    df['ts'] = pd.to_datetime(df['ts'])
    groups = df.groupby('caseid')
    encoded_df=[]
    for case,group in groups: 
        group = group.reset_index(drop=True)
        if len(group)>prefix:
            group = group.loc[:prefix-1,:]
            encoded_df.append(group)
    return pd.concat(encoded_df)

def indexbase_encoding(df, prefix):
    '''
    Indexbase encoding
    
    Parameters
    ----------
    df : pandas dataframe
        Assigned dataframe to encode for outcome prediction
    
    prefix : int
        Prefix length to slice to cases in fixed length
    
    Returns
    ----------
    Return dataframe encoded in indexbase method
    '''
    df = filter_by_prefix(df,prefix)
    df['ts'] = pd.to_datetime(df['ts'])
    groups = df.groupby('caseid')
    encoded_df=[]
    if 'resource' not in list(df.columns.values):
        noresource = True
    else:
        noresource = False
        
    for case,group in groups: 
        activitylist = list(group['activity'])
        
        group = group.reset_index(drop=True)
        outcome = set(group['outcome']).pop()
        cumdurationlist = [(x - list(group['ts'])[0]).total_seconds() for x in list(group['ts'])]
        cumduration_index ={'Cumduration_'+str(x+1): cumdurationlist[x] for x in range(len(cumdurationlist))}
        
        case_outcome = {'caseid':case, 'outcome':outcome}
        activity_index = {'activity_'+str(x+1)+'_'+activitylist[x]: 1 for x in range(len(activitylist))}

        if noresource == False:
            resourcelist = list(group['resource'])
            resource_index = {'resource_'+str(x+1)+'_'+str(resourcelist[x]): 1 for x in range(len(resourcelist))}
            case_outcome.update(resource_index)
        
        case_outcome.update(cumduration_index)
        case_outcome.update(activity_index)
        dfk = pd.DataFrame.from_dict([case_outcome])
        encoded_df.append(dfk)
    concated_df = pd.concat(encoded_df)
    concated_df = concated_df.fillna(0)
    return concated_df

#%%
df = pd.read_csv('./preprocessed_loan_baseline.pnml_noise_0.049999999999999996_iteration_1_seed_42477_sample.csv')


key_pair = {'Case ID':'caseid', 'Activity':'activity', 'Complete Timestamp':'ts'}
df = df.rename(columns=key_pair)

if 'resource' in df.columns.values:
    df = df.loc[:,['caseid','activity','ts','resource','noise']]

else:
    df = df.loc[:,['caseid','activity','ts','noise']]

# try:
#     os.makedirs('./result/%s'%(dataset_label))
# except:
#     pass

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

max_case_len =10
idslist = []
for prefix in range(1, max_case_len):
    idslist.append(indexbase_encoding(dfn,prefix))

prefixlist= list(range(1, max_case_len))
acc_dict= {}


#%%
##############################
#Next activity prediction part
##############################
print('Random forest')
models = []
used_models = 'RF'
testdf_list = []

for pos,prefix in enumerate(idslist):
    np.random.seed(2022)
    trainids = np.random.choice(caseids, int(len(caseids)*0.7), replace=False)

    traindf = prefix[prefix['caseid'].isin(trainids)].reset_index(drop=True)
    testdf = prefix[~prefix['caseid'].isin(trainids)].reset_index(drop=True)
    testdf_list.append(testdf)

    y_train = traindf['outcome']
    x_train = traindf.drop(columns=['outcome','caseid'],axis=1)

    y_test = testdf['outcome']
    x_test = testdf.drop(columns=['outcome','caseid'],axis=1)

    # Random forest result    
    
    rf = RandomForestClassifier(criterion='entropy').fit(x_train,y_train)
    y_pred = rf.predict(x_test)

    filename = './models/%s prefix %s.pkl'%(used_models, pos+1)
    models.append(rf)
    with open(filename,'wb') as f:
        pkl.dump(rf, f)

    acc_dict['prefix_%s'%(str(prefixlist[pos]))] =  accuracy_score(y_test,y_pred)

#%%
# testing_case = testdf_list[-1].loc[1,:]
# y_testing_case = testing_case['outcome']
# x_testing_case = testing_case.drop(labels=['outcome','caseid'])

# x_testing_case = np.array(x_testing_case.values).reshape(1,-1)
# predicted = models[-1].predict_proba(x_testing_case)
# print(testing_case, predicted)
# print(models[-1].classes_[np.argmax(predicted)])


testing_case_ids = set(testdf_list[-1]['caseid'])
testdf = df[df['caseid'].isin(testing_case_ids)]
print(testdf)

#%%
model = models[-1]
for_confusion_matrix = {}

counting_normal = 0

for threshold in [0.01,0.05,0.1,0.15,0.2,0.25]:
    global_true =[]
    global_pred = []

    for caseid in list(testing_case_ids):

        for_confusion_matrix[int(caseid)] =[]

        prediction_list = []

        df = testdf
        for pos, prefix in enumerate(idslist):
            prediction_label = 'Normal'

            x_test = testdf_list[pos][testdf_list[pos]['caseid'] ==caseid]
            true_outcome = x_test['outcome']
            print(x_test.columns.values)
            x_test.drop(labels= ['caseid', 'outcome'])
            print(true_outcome)
            x_test = np.array(x_test.values).reshape(1,-1)
            print(x_test)
    #         predictions = models[pos].predict()
    #         predictions_proba = predictions[0][0]
    #         predictions_value = list(predictions[1])
    #         if predictions  == 'Not Available':
    #             prediction_label = 'Not Available'
    #         else:
    #             if t.true_label in predictions_value:
    #                 labelidx = predictions_value.index(t.true_label)

    #                 if predictions_proba[labelidx] <threshold:
    #                     prediction_label = 'Anomalous'
    #             else:
    #                 prediction_label = 'Anomalous'

    #         if t.true_label != 'End':
    #             prediction_list.append(prediction_label)


    #     true_label_list = []

    #     labellist = list(df['noise'])
    #     actlist = list(df['Activity'])
    #     for pos, t in enumerate(labellist):
    #         if t == 'Start' or t == 'End':
    #             continue
    #         elif t == 'true':
    #             true_label = 'Anomalous'
    #         else:
    #             true_label = 'Normal'
    #         true_label_list.append(true_label)


    #     for pos, p in enumerate(prediction_list):
    #         global_pred.append(p)
    #         global_true.append(true_label_list[pos])


    # saving_data = {'y_true':global_true, 'y_pred':global_pred}