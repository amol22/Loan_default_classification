import sys
import os
import dill
import numpy as np
import pandas as pd
from typing import Tuple
from src.exception import CustomException
from sklearn.metrics import precision_score, recall_score, f1_score

def save_object(file_path:str,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_test,y_test,model):
    try:
        y_pred = model.predict(X_test)
        y_pred_binary = np.round(y_pred)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        return f1
    except Exception as e:
        raise CustomException(e,sys)