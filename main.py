from src.data_preprocessing import data_preprocessing
from src.data_ingestion import data_ingestion 
def main():
    df = data_ingestion()
    print(df.columns)
    X_train, X_test, y_train, y_test = data_preprocessing(df)
    print(X_train)
main()

