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

from cmath import exp
import dice_ml
from dice_ml import Dice

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier as rfcls

########################
### Data preparation ###
########################

df = pd.read_csv('./data/Prefix 5 dataset.csv')

used_models = 'RF'

key_pair = {'Case ID':'caseid', 'Activity':'activity', 'Complete Timestamp':'ts'}
df = df.rename(columns=key_pair)

if 'resource' in df.columns.values:
    df = df.loc[:,['caseid','activity','ts','resource','outcome','noise']]

else:
    df = df.loc[:,['caseid','activity','ts','noise', 'outcome']]

groups = df.groupby('caseid')
concating = []

outcome_dict = {x: pos+1 for pos, x in enumerate(list(set(df['outcome'])))}
new_outcome = []
for x in list(df['outcome']):
    new_outcome.append(outcome_dict[x])

df['outcome'] =new_outcome


outcome = []
for _, group in groups:
    group['ts'] = pd.to_datetime(group['ts'])
    group = group.sort_values(by='ts')
    group = group.reset_index(drop=True)
    case_length = len(group)
    prep_group = {'caseid': list(group['caseid'])[-1], 'outcome': list(group['outcome'])[-1]}
    
    ####################
    ###Activity label###
    ####################
    actlist = list(group['activity'])
    new_actcolumns =['Activity_%s'%(x+1) for x in range(case_length)]
    for pos, n in enumerate(new_actcolumns):
        prep_group[n] = actlist[pos]
    

    ####################
    ###  Timestamp   ###
    ####################

    durationlist = []
    for pos, x in enumerate(list(group['ts'])):
        if pos ==0:
            durationlist.append(0)
        else:
            durationlist.append((x - list(group['ts'])[pos-1]).total_seconds())
    duration_index ={'Duration_%s'%(x+1): durationlist[x] for x in range(len(durationlist))}
    cumdurationlist = [(x - list(group['ts'])[0]).total_seconds() for x in list(group['ts'])]
    cumduration_index ={'Cumduration_'+str(x+1): cumdurationlist[x] for x in range(len(cumdurationlist))}

    prep_group.update(duration_index)
    prep_group.update(cumduration_index)

    ####################
    ###   Resource   ###
    ####################
    
    if 'resource' in group.columns.values:
        reslist = list(group['resource'])
        new_resourceolumns = ['Resource_%s'%(x+1) for x in range(case_length)]
        for pos, n in enumerate(new_resourceolumns):
            prep_group[n] = reslist[pos]
    
    ####################
    ###  Concating   ###
    ####################
#     print(prep_group)
#     print(pd.DataFrame.from_dict(prep_group))

    concating.append(prep_group)

dfn = pd.DataFrame.from_dict(concating)

datasetX = dfn.drop(columns=['caseid'],axis=1)

outcome_name = 'outcome'
categorical_features =[]
continuous_features = []


for x in datasetX.columns.values:
    if x =='caseid' or x==outcome_name:
        pass
    elif datasetX[x].dtype == 'object':
        categorical_features.append(x)
    else:
        continuous_features.append(x)


target = datasetX[outcome_name]
print(target)
datasetXY = datasetX.drop(outcome_name,axis=1)

x_train, x_test, y_train, y_test = train_test_split(datasetXY,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, continuous_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', transformations),
                           ('classifier', rfcls(n_estimators = 10))])
model = clf.fit(x_train, y_train)

d = dice_ml.Data(dataframe=datasetX,
                  continuous_features=continuous_features,
                  outcome_name=outcome_name)

# We provide the type of model as a parameter (model_type)
m = dice_ml.Model(model=model, backend="sklearn", model_type='classifier')

exp_genetic = Dice(d, m, method="genetic")

# Single input
query_instances = x_test[8:15]
prediction = model.predict(query_instances)[0]

actual_activity_label = [activity for activity,act_num in outcome_dict.items() if act_num == prediction][0]
wanted_activity_label = [activity for activity,act_num in outcome_dict.items() if act_num == 16][0]
print(actual_activity_label)
print(wanted_activity_label)
print(model.predict_proba(query_instances))
print(model.classes_)

# features_to_vary =categorical_features + [x for x in continuous_features if 'Duration' in x]
# features_to_vary.remove('Duration_1')
# features_to_vary.remove('Activity_1')
# print(features_to_vary)
# genetic_iris = exp_genetic.generate_counterfactuals(query_instances, total_CFs=3, desired_class=16, features_to_vary=features_to_vary)
# results = genetic_iris.visualize_as_dataframe(show_only_changes=False)
# results