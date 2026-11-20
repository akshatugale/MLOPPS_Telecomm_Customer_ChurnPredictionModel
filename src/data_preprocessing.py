from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer,KNNImputer
from imblearn.over_sampling import SMOTE

def data_preprocessing(df):


    #step1 : drop the duplicated values
    #step2 : Feature Engineering (optional)
    #step3 : split data into X and y 
    #step4 : split the ddataset into train and test 
    #step5 : Segregate the data into categorical and numerical columns
    #step6 : Build numerical_pipeline : imputatuion, scaling
    #step7 : Build categorical_pipeline : imputation, encoding
    #step8 : Concatenate the numerical and categorical pipeline using ColumnTransformer
    #step9 : Fit the transformer on the train and test
    #step10 : fit_resample the data so that the target column is balanced (only for classsification)
    #step11 : return the preprocessed X_train , X_test, y_train, y_test
    #---------------------------------------------------------------------------------------------------------------------

    #step1 : drop the duplicated values
    df.drop_duplicates(inplace=True)

    #step2 : Feature Engineering (optional)

    #step3 : split data into X and y    

    X = df.drop(columns=['customerID','Churn'])
    y = df['Churn']

    #step4 : split the ddataset into train and test 

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

    #step5 : Segregate the data into categorical and numerical columns

    numerical_col = X_train.select_dtypes(exclude='object').columns
    categorical_col = X_train.select_dtypes(include='object').columns

    #step6 : Build numerical_pipeline : imputatuion, scaling

    numerical_pipeline = Pipeline([('imputation', SimpleImputer(strategy='median')), ('scaling', MinMaxScaler())])

    #step7 : Build categorical_pipeline : imputation, encoding
    categorical_pipeline = Pipeline([('imputation', SimpleImputer(strategy='most_frequent')), ('encoding', OneHotEncoder(handle_unknown='ignore'))])

    #step8 : Concatenate the numerical and categorical pipeline using ColumnTransformer
        
    transformer = ColumnTransformer([('numerical_pipeline', numerical_pipeline, numerical_col), ('categorical_pipeline', categorical_pipeline, categorical_col)])

    #step9 : Fit the transformer on the train and test

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    #step10 : fit_resample the data so that the target column is balanced (only for classsification)

    ''' 
    *generally used with classification model
    *target column balance technique basically classified into two...
        - over_sampling technique
        - under_sampling technique

    '''

    X_train_resample, y_train_resample = SMOTE().fit_resample(X_train, y_train)

    #step11 : return the preprocessed X_train , X_test, y_train, y_test
    return X_train, X_test, y_train, y_test