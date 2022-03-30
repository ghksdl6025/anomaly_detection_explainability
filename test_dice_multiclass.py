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


def iris_multiclass():
    df_iris = load_iris(as_frame=True).frame

    outcome_name = 'target'
    continuous_features_iris = df_iris.drop(outcome_name, axis=1).columns.tolist()
    target = df_iris[outcome_name]

    datasetX = df_iris.drop(outcome_name, axis=1)
    x_train,x_test,y_train,y_test = train_test_split(datasetX,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)

    categorical_features = x_train.columns.difference(continuous_features_iris)

    numeric_transformer = Pipeline(steps=[
                                ('scaler', StandardScaler())
                                ])

    categorical_transformer = Pipeline(steps=[
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                ])
    transformations = ColumnTransformer(
        transformers = [
                        ('num', numeric_transformer, continuous_features_iris),
                        ('cat', categorical_transformer, categorical_features)
                        ])

    clf_iris = Pipeline(steps=[
                        ('preprocessor', transformations),
                        ('classifier', rfcls())
                        ])

    model_iris = clf_iris.fit(x_train, y_train)

    d_iris = dice_ml.Data(dataframe=df_iris,
                        continuous_features=continuous_features_iris,
                        outcome_name=outcome_name)

    m_iris = dice_ml.Model(model=model_iris, backend='sklearn', model_type='classifier')

    exp_genetic_iris = Dice(d_iris, m_iris, method='genetic')

    query_instance_iris = x_test[2:3]
    genetic_iris = exp_genetic_iris.generate_counterfactuals(query_instance_iris, total_CFs=7, desired_class=2)
    genetic_iris.visualize_as_dataframe()

    query_instance_iris = x_test[7:9]
    genetic_iris = exp_genetic_iris.generate_counterfactuals(query_instance_iris, total_CFs=7, desired_class=0)
    genetic_iris.visualize_as_dataframe(show_only_changes=True)


housing_data = fetch_california_housing()
outcome_name = 'target'
df_housing = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
df_housing[outcome_name] = pd.Series(housing_data.target)

continuous_features_housing = df_housing.drop(outcome_name, axis=1).columns.tolist()
target = df_housing[outcome_name]

datasetX = df_housing.drop(outcome_name, axis=1)
x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                target,
                                                test_size=0.2,
                                                random_state=0)

categorical_features = x_train.columns.difference(continuous_features_housing)

numeric_transformer = Pipeline(steps=[
                            ('scaler', StandardScaler())
                            ])

categorical_transformer = Pipeline(steps=[
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                ])

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, continuous_features_housing),
        ('cat', categorical_transformer, categorical_features)
        ])

regr_housing = Pipeline(steps=[
                        ('preprocessor', transformations),
                        ('regressor', rfreg())
                        ])

model_housing = regr_housing.fit(x_train, y_train)

d_housing = dice_ml.Data(dataframe = df_housing,
                        continuous_features = continuous_features_housing,
                        outcome_name=outcome_name)
m_housing = dice_ml.Model(model=model_housing, backend='sklearn', model_type='regressor')

exp_genetic_housing = Dice(d_housing, m_housing, method='genetic')

query_instance_housing = x_test[2:4]
genetic_housing = exp_genetic_housing.generate_counterfactuals(
                                query_instance_housing,
                                total_CFs=3,
                                desired_range=[3.0,5.0])
genetic_housing.visualize_as_dataframe(show_only_changes=True)

































