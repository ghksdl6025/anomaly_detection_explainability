import dice_ml
from dice_ml.utils import helpers
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import tensorflow as tf

import pandas as pd
import numpy as np

#%% Data reading
file = open('./adult.data','r')
raw_data = np.genfromtxt(file, delimiter=', ', dtype=str, invalid_raise=False)

#  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'income']

adult_data = pd.DataFrame(raw_data, columns=column_names)

# For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
adult_data = adult_data.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})

adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government',
                                    'Local-gov': 'Government'}})
adult_data = adult_data.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
adult_data = adult_data.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

adult_data = adult_data.replace(
    {
        'occupation': {
            'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
            'Exec-managerial': 'White-Collar', 'Farming-fishing': 'Blue-Collar',
            'Handlers-cleaners': 'Blue-Collar',
            'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service',
            'Priv-house-serv': 'Service',
            'Prof-specialty': 'Professional', 'Protective-serv': 'Service',
            'Tech-support': 'Service',
            'Transport-moving': 'Blue-Collar', 'Unknown': 'Other/Unknown',
            'Armed-Forces': 'Other/Unknown', '?': 'Other/Unknown'
        }
    }
)

adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
                                                    'Married-spouse-absent': 'Married', 'Never-married': 'Single'}})

adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                            'Amer-Indian-Eskimo': 'Other'}})

adult_data = adult_data[['age', 'workclass', 'education', 'marital-status', 'occupation',
                            'race', 'gender', 'hours-per-week', 'income']]

adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                                '11th': 'School', '10th': 'School', '7th-8th': 'School',
                                                '9th': 'School', '12th': 'School', '5th-6th': 'School',
                                                '1st-4th': 'School', 'Preschool': 'School'}})

adult_data = adult_data.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

#%%

target = adult_data['income']

train_dataset, test_dataset, y_train, y_test = train_test_split(adult_data,target, test_size=0.2, stratify=target)

x_train = train_dataset.drop('income', axis=1)
x_test = train_dataset.drop('income', axis=1)

d = dice_ml.Data(dataframe=train_dataset,
                 continuous_features=['age', 'hours_per_week'],
                 outcome_name='income')

numerical = ['age', 'hours_per_week']
categorical = x_train.columns.difference(numerical)


categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
transformations = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical)])

clf = Pipeline(steps=[('preprocessing', transformations),
                ('classifier',rf(max_depth=10))])

model = clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

m = dice_ml.Model(model,backend='sklearn')
exp = dice_ml.Dice(d,m, method='random')

query_instance = {'age':22,
    'workclass':'Private',
    'education':'HS-grad',
    'marital_status':'Single',
    'occupation':'Service',
    'race': 'White',
    'gender':'Female',
    'hours_per_week': 45}

dice_exp = exp.generate_counterfactuals(x_test[0:1], total_CFs=4, desired_class='opposite', features_to_vary=['workclass','education','occupation','hours_per_week'])
dice_exp.visualize_as_dataframe(show_only_changes=False)