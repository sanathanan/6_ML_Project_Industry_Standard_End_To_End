import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    """
    Sending the parameters from data transformation to this function for
    model training.
    Providing the train and test data for model training operation.
    """
    def initiate_model_trainer(self,train_arr, test_arr):
        try:
            logging.info("Splitting training and test input data")
            # We giving train and test data input and output feature.
            # i.e, Normally, we give X = df[input_features], y=df[output_feature], test_size=0.2) which will take data for train and test 
            # automaticaly. But here, we divided the dat.csv into train.csv and test.csv. so giving like below.
            X_train, Y_train, X_test, Y_test = (train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1])

            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                # "XGBRegressor": XGBRegressor(), 
                # "K-Neighbors Regressor": KNeighborsRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Hyper Parameter Tuning Parameters
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                # "XGBRegressor":{
                #     'learning_rate':[.1,.01,.05,.001],
                #     'n_estimators': [8,16,32,64,128,256]
                # },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # We are getting the model report in the form of dictionary
            model_report:dict = evaluate_model(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, models=models, params=params)

            # To get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            # Based on the 'best_model_score' we are getting the 'best_model_name' from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Finding the model name from models
            best_model = models[best_model_name]

            # Validation logic
            if best_model_score < 0.6:
                raise CustomException('No best model found')
            logging.info("Best found model on both training and testing dataset")

            # Saving the file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            # Precition testing
            predicted = best_model.predict(X_test)
            r2_square = r2_score(Y_test, predicted)
            
            return r2_square
            

        except Exception as e:
            raise CustomException(e, sys)
