import os
import sys
from src.exception import CustomException
from src.logger import logging
from config import ModelTrainerConfig
from src.utils import save_object
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from src.utils import evaluate_model

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Train and test arrays split")
            
            model = Sequential()
            model.add(Dense(32, input_shape=(16,), activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            logging.info("Model training started")
            model.fit(X_train, y_train, epochs=50)
            f1 = evaluate_model(X_test,y_test,model)
            if f1<0.7:
                raise CustomException("F1 score too low")
            logging.info("Model training complete")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=model)       
        except Exception as e:
            raise CustomException(e,sys)


