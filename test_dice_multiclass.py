#%%
from cmath import exp
import dice_ml
from dice_ml import Dice

from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier as rfcls
from sklearn.ensemble import RandomForestRegressor as rfreg

from xgboost import XGBClassifier as xgb

import pandas as pd
import numpy as np

#%%
df_iris = load_iris(as_frame=True).frame
df_iris.head()
outcome_name = "target"

random_col =[]

for x in range(len(df_iris)):
    random_col.append(np.random.choice(['A','B','C']))
df_iris['Random_col'] = random_col
continuous_features_iris = df_iris.drop(outcome_name, axis=1).columns.tolist()
continuous_features_iris.remove('Random_col')
target = df_iris[outcome_name]

# Split data into train and test
datasetX = df_iris.drop(outcome_name, axis=1)
x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)

categorical_features = x_train.columns.difference(continuous_features_iris)
print(categorical_features)


#%%
# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, continuous_features_iris),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf_iris = Pipeline(steps=[('preprocessor', transformations),
                           ('classifier', rfcls())])
model_iris = clf_iris.fit(x_train, y_train)

d_iris = dice_ml.Data(dataframe=df_iris,
                      continuous_features=continuous_features_iris,
                      outcome_name=outcome_name)

# We provide the type of model as a parameter (model_type)
m_iris = dice_ml.Model(model=model_iris, backend="sklearn", model_type='classifier')

exp_genetic_iris = Dice(d_iris, m_iris, method="genetic")

# Single input
query_instances_iris = x_test[2:3]
genetic_iris = exp_genetic_iris.generate_counterfactuals(query_instances_iris, total_CFs=7, desired_class=2)
genetic_iris.visualize_as_dataframe()
# %%
