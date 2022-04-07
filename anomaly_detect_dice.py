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

datasetX = idslist[-1]

outcome_name = 'outcome'
target = datasetX[outcome_name]
categorical_features =[]
continuous_features = []

for x in datasetX.columns.values:
    if 'Cum' in x:
        continuous_features.append(x)
    elif x =='caseid' or x ==outcome:
        pass
    else:
        categorical_features.append(x)


np.random.seed(0)
caseids = list(set(datasetX['caseid']))
trainids = np.random.choice(caseids, int(len(caseids)*0.7), replace=False)
traindf = datasetX[datasetX['caseid'].isin(trainids)].reset_index(drop=True)
testdf = datasetX[~datasetX['caseid'].isin(trainids)].reset_index(drop=True)


y_train = traindf[outcome_name]
x_train = traindf.drop(columns=[outcome_name,'caseid'],axis=1)

y_test = testdf[outcome_name]
test_ids = set(testdf['caseid'])
x_test = testdf.drop(columns=[outcome_name,'caseid'],axis=1)


regr_housing = Pipeline(steps=[
                        ('classifier', rfcls())
                        ])

model_housing = regr_housing.fit(x_train, y_train)

datasetXy = datasetX.drop(columns=['caseid'], axis=1)

d_housing = dice_ml.Data(dataframe = datasetXy,
                        continuous_features = continuous_features,
                        outcome_name=outcome_name)
m_housing = dice_ml.Model(model=model_housing, backend='sklearn', model_type='classifier')

exp_genetic_housing = Dice(d_housing, m_housing, method='genetic')

test_df = datasetX[datasetX['caseid'].isin(test_ids)].sort_values(by='caseid')


test_df = pd.read_csv('./testdf.csv')
print(test_ids)
query_instance_housing = test_df.iloc[3,:]
print(query_instance_housing)
test_outcome = query_instance_housing[outcome_name].values
query_instance_housing = query_instance_housing.drop(columns=['caseid', outcome_name], axis=1)
predicted_one = model_housing.predict(query_instance_housing)
model_classes = model_housing.classes_
predicted_proba = model_housing.predict_proba(query_instance_housing)
print(predicted_one, predicted_proba, test_outcome, list(model_classes).index(test_outcome[0]))

# genetic_housing = exp_genetic_housing.generate_counterfactuals(
#                                 query_instance_housing,
#                                 total_CFs=3,
#                                 desired_range=[,1])
# genetic_housing.visualize_as_dataframe(show_only_changes=True)

