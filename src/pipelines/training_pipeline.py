import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.config import DataIngestionConfig, DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

if __name__=="__main__":
    obj = DataIngestion()
    train_path,test_path = obj.data_ingestion()
    data_transform = DataTransformation()
    train_arr,test_arr = data_transform.initiate_data_transformation(train_path,test_path)
    model = ModelTrainer()
    model.initiate_model_trainer(train_arr,test_arr)