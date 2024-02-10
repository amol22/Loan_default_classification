import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            processed_data = preprocessor.transform(features)
            pred = model.predict(processed_data)
            pred_bin = np.round(pred)
            return pred_bin
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,
                 Age: int,
                 Income: int,
                 LoanAmount: int,
                 CreditScore: int,
                 MonthsEmployed: int,
                 NumCreditLines: int,
                 InterestRate: float,
                 LoanTerm: int,
                 DTIRatio: float,
                 Education: str,
                 EmploymentType: str,
                 MaritalStatus: str,
                 HasMortgage: str,
                 HasDependents: str,
                 LoanPurpose: str,
                 HasCoSigner: str):
        self.age = Age
        self.income = Income
        self.loanamount = LoanAmount
        self.creditscore= CreditScore
        self.monthsemployed = MonthsEmployed
        self.numcreditlines = NumCreditLines
        self.interestrate = InterestRate
        self.loanterm = LoanTerm
        self.dtiratio = DTIRatio
        self.education = Education
        self.employmenttype = EmploymentType
        self.maritalstatus = MaritalStatus
        self.hasmortgage = HasMortgage
        self.hasdependents = HasDependents
        self.loanpurpose = LoanPurpose
        self.hascosigner = HasCoSigner
        
    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                "Age": [self.age],
                "Income": [self.income],
                "LoanAmount": [self.loanamount],
                "CreditScore": [self.creditscore],
                "MonthsEmployed": [self.monthsemployed],
                "NumCreditLines": [self.numcreditlines],
                "InterestRate": [self.interestrate],
                "LoanTerm": [self.loanterm],
                "DTIRatio": [self.dtiratio],
                "Education": [self.education],
                "EmploymentType": [self.employmenttype],
                "MaritalStatus":[self.maritalstatus],
                "HasMortgage":[self.hasmortgage],
                "HasDependents":[self.hasdependents],
                "LoanPurpose":[self.loanpurpose],
                "HasCoSigner":[self.hascosigner]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)