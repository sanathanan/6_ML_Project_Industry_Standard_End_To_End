import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer # For pipeline configuration for one hot encoding, standard scaling and so on.
from sklearn.impute import SimpleImputer # For work on missing values
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

"""
In Data_Transformation, if we required any input (i.e path for model storage and so on). 
We will give through this 'DataTransformationCofig' class.
"""
@dataclass
class DataTransformationCofig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationCofig()

    """
    This function will convert categorical feature into numerical feature,
    Standard scaling and so on.
    Basically, this function is responsible for data transformation.
    """
    def get_data_transformer_object(self):
        logging.info("Entered the get_data_transformer_object method")
        try:
            # Numerical Features present in the dataset
            numerical_columns = ['reading_score', 'writing_score']
            logging.info(f"(Numerical columns: {numerical_columns})")

            # Categorical Features present in the dataset
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            logging.info(f"(Categorical columns: {categorical_columns})")

            # Numerical pipeline for handling missing values and scaling the data.
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')), # Handling Missing values
                    ('scaler', StandardScaler()) # Scaling the data
                ]
            )
            logging.info('Numerical columns standard scaling completed')
            
            # Categorical pipeline for handling missing values, convertint categorical
            # features into numerical features and scaling down the data.
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')), # Handling missing values
                    ('one_hot_encoder', OneHotEncoder()), # Converting categorical features into numerical features
                    ('scaler', StandardScaler(with_mean=False)) # Scaling the data
                ]
            ) 
            logging.info('Categorcial columns encoding completed')

            # We need to combine numrical pipeline with categorical pipeline
            # 'ColumnTransformer' will combine things.
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    
    def initiate_data_transformation(self, train_data_path, test_data_path):
        logging.info("Entered the initiate_data_transformation method")
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)



