'''
Store necessary variables for pytest

Author: Luiz Coelho
March 2023
'''

import pytest

def df_plugin():
    '''Placeholder function'''
    return None

# Creating a Dataframe object 'pytest.df' in Namespace
def pytest_configure():
    '''Store testing variables'''
    pytest.df = None
    pytest.X_train_df = None
    pytest.X_test_df = None
    pytest.y_train_df = None
    pytest.y_test_df = None
