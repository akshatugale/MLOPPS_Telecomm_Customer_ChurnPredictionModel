import pandas as pd 

def data_ingestion():
    '''
    *TItle : 
    *independent column [X] : 
    *Target column [y] :
    *Problem Statement:
    1. to predict the churn rate 
    2. use clustering approach to predict and classify the cluster pattern such as {l1,l2,l3}

    '''
    df = pd.read_csv(r'C:\MLOPPS_Telecomm_Customer_ChurnPredictionModel\data\raw\churn - churn.csv')

    return df