import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from src.exception import CustomException
from src.logger import logging
from config import DataTransformationConfig
from src.utils import save_object


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer(self):
        """Initiates and gets the data transformer object

        Raises:
            CustomException: exception
        """
        try:
            # Quantitative variables:
            num_cols = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
            "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]
            # Categorical variables:
            cat_cols = ["Education", "EmploymentType", "MaritalStatus",
            "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]
            
            num_pipeline = Pipeline(
                steps=[("standard_scaler",StandardScaler())]
            )
            
            cat_pipeline = Pipeline(
                steps=[("target_encoder",TargetEncoder(min_samples_leaf = 1, smoothing = 10))]
            )
            
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_cols),
                ("cat_pipeline",cat_pipeline,cat_cols)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            train_df.drop("LoanID", axis = 1, inplace = True)
            test_df.drop("LoanID", axis = 1, inplace = True)
            
            preprocessor = self.get_data_transformer()
            logging.info("Preprocessor object obtained")
            
            target_column_name = 'Default'
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("applying preprocessor")
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df,target_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                obj = preprocessor
                )
            
            logging.info("Preprocessor object trained and stored")
            
            return (train_arr,test_arr)
            
        except Exception as e:
            raise CustomException(e,sys)
        
        