'''
Tests module for churn_library.py

Author: Luiz Coelho
March 2023
'''

import os
import logging
import pytest
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
    '''
    test data import - this example is completed for you to assist with the 
                       other test functions
    '''
    try:
        dataframe = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows"
                      "and columns")
        raise err
    pytest.df = dataframe

def test_eda():
    '''
    test perform eda function
    '''
    dataframe = pytest.df
    try:
        cls.perform_eda(dataframe)
    except NameError as err:
        logging.error("Variable df not defined")
        raise err

    expected_files = ["./images/eda/churn_hist.png",
                      "./images/eda/marital_status_hist.png", 
                      "./images/eda/heatmap.png"]
    for filename in expected_files:
        try:
            assert os.path.exists(filename)
        except AssertionError as err:
            logging.error("File was not created: %s", filename)
            raise err
    pytest.df = dataframe
    logging.info("Testing perform_eda: SUCCESS")

def test_encoder_helper():
    '''
    test encoder helper
    '''
    dataframe = pytest.df
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
    ]
    try:
        dataframe = cls.encoder_helper(dataframe, cat_columns)
    except NameError as err:
        logging.error("Variable df not defined")
        raise err

    new_col_name_list = [col_name + '_Churn' for col_name in cat_columns]
    for new_col_name in new_col_name_list:
        try:
            assert new_col_name in dataframe
        except AssertionError as err:
            logging.error("Df does not have column: %s", new_col_name)
            raise err

    pytest.df = dataframe
    logging.info("Testing encoder_helper: SUCCESS")



def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    dataframe = pytest.df
    try:
        x_train_df, x_test_df, y_train_df, y_test_df = cls.perform_feature_engineering(dataframe)
    except NameError as err:
        logging.error("Variable df not defined")
        raise err

    try:
        assert len(x_train_df.shape) == 2
        assert len(x_test_df.shape) == 2
        assert len(y_train_df.shape) == 1
        assert len(y_test_df.shape) == 1
    except AssertionError as err:
        logging.error("Train or test df have wrong shape")
        raise err

    pytest.X_train_df = x_train_df
    pytest.X_test_df = x_test_df
    pytest.y_train_df = y_train_df
    pytest.y_test_df = y_test_df
    logging.info("Testing feature_engineering: SUCCESS")


def test_train_models():
    '''
    test train_models
    '''
    x_train_df = pytest.X_train_df
    x_test_df = pytest.X_test_df
    y_train_df = pytest.y_train_df
    y_test_df = pytest.y_test_df
    try:
        cls.train_models(x_train_df, x_test_df, y_train_df, y_test_df)
    except NameError as err:
        logging.error("Variables train_df not defined")
        raise err

    try:
        assert os.path.exists("./models/logistic_model.pkl")
        assert os.path.exists("./models/rfc_model.pkl")
    except AssertionError as err:
        logging.error("Trained models were note saved")
        raise err

    logging.info("Testing train_models: SUCCESS")

if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
