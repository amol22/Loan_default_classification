import streamlit as st
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


def main():
    st.title("Loan Default Prediction")
    st.write("Enter the details below")
    col1,col2 = st.columns(2)
    age = col1.number_input("Age",min_value=1,max_value=130)
    income = col2.number_input("Income", min_value =1, max_value=500000000)
    loanamount = col1.number_input("Loan Amount", min_value=1, max_value=1000000000)
    creditscore = col2.number_input("Credit Score", min_value=1, max_value=1500)
    monthsemp = col1.number_input("Months employed", min_value=1, max_value=570)
    numcredlines = col2.number_input("Number of Credit Lines", min_value=0, max_value=50)
    interestrate = col1.number_input("Interest Rate", min_value=1.0, max_value=15.0, step=0.5)
    loanterm = col2.number_input("Loan Term(in months)",min_value=1, max_value=360)
    dtiratio = col1.number_input("DTI Ratio", min_value=0.0, max_value=15.0)
    education = st.selectbox("Education", ('High School','Bachelor\'s','Master\'s','PHD'))
    employmenttype = st.selectbox("Employment Type", ('Unemployed','Part-time','Full-time','Self-employed'))
    maritalstatus = st.selectbox("Marital Status", ('Single','Married','Divorced'))
    loanpurpose = st.selectbox("Loan Purpose", ('Business','Home','Education','Auto','Other'))
    hasmortgage = st.selectbox("Mortgage", ('Yes','No'))
    hasdependents = st.selectbox("Dependents", ('Yes','No'))
    hascosigner = st.selectbox("CoSigner", ('Yes','No'))
    
    if st.button('Predict'):
        data = CustomData(age,income,loanamount,creditscore,monthsemp,numcredlines,
                          interestrate,loanterm,dtiratio,education,employmenttype,maritalstatus,
                          hasmortgage,hasdependents,loanpurpose,hascosigner)
        pred_df = data.get_data_as_df()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        if results == 1:
            st.success('The loan will default')
        else:
            st.success('The loan will not default')
            
        
    
    
    
    
if __name__ == "__main__":
    main()