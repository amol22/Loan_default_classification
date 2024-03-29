import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from config import DataIngestionConfig

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def data_ingestion(self):
        """Ingests the data into a dataframe and performs a train-test split
        """
        logging.info("Data ingestion initiated")
        try:
            df = pd.read_csv("data/Loan_default.csv")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok  = True)
            #df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            
            logging.info("Train-test split initiated")
            df.drop("LoanID", axis = 1, inplace = True)
            X = df.drop("Default", axis = 1)
            y = df["Default"]
            smote = SMOTENC(categorical_features=[9,10,11,12,13,14,15], sampling_strategy="minority")
            X_sm, y_sm = smote.fit_resample(X,y)
            X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.2, random_state = 22, stratify= y_sm)
            train_set = pd.concat([X_train,y_train], axis=1)
            test_set = pd.concat([X_test,y_test], axis=1)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data ingestion completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
        except Exception as e:
            raise CustomException(e,sys)