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


df = pd.read_csv('./preprocessed_loan_baseline.pnml_noise_0.049999999999999996_iteration_1_seed_42477_sample.csv')


key_pair = {'Case ID':'caseid', 'Activity':'activity', 'Complete Timestamp':'ts'}
df = df.rename(columns=key_pair)

if 'resource' in df.columns.values:
    df = df.loc[:,['caseid','activity','ts','resource','noise']]

else:
    df = df.loc[:,['caseid','activity','ts','noise']]

try:
    os.makedirs('./result/%s'%(dataset_label))
except:
    pass


caseids = list(set(df['caseid']))
np.random.seed(2022)
trainids = np.random.choice(caseids, int(len(caseids)*0.7), replace=False)
traindf = df[df['caseid'].isin(trainids)].reset_index(drop=True)
testdf = df[~df['caseid'].isin(trainids)].reset_index(drop=True)


print(traindf)


groups = df.groupby('caseid')
concating = []
for _, group in groups:
    outcomelist = list(group['outcome'])
    outcome = outcomelist[-1]
    group = group.reset_index(drop=True)
    if True in outcomelist:
        group = group.loc[:outcomelist.index(True),:]
    group['outcome'] = outcome
    concating.append(group)

dfn = pd.concat(concating)