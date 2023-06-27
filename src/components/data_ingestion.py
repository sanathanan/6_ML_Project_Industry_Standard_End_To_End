import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


"""
In Data_Ingestion, if we required any input (i.e path for training_data, test_data, raw_data and so on). 
We will give through this 'Data_Ingestion class'.
"""
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "data.csv")

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    """
    Reading data from source location.
    """
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Reading the data from storage
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset from the dataframe")

            # Making directories as defined in "DataIngestionConfig"
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            #  Creating csv file in 'raw_data_path' location
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Splitting the dataset for training and testing
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            #  Creating csv file in 'train_data_path' location
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            #  Creating csv file in 'test_data_path' location
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()