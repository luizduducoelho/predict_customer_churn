# library doc string
'''
Churn library with necessary functions

Author: Luiz Coelho
March 2023
'''

# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, RocCurveDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

sns.set()
os.environ['QT_QPA_PLATFORM']='offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    save_path = "images/eda/"
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(lambda val: 0
                                       if val == "Existing Customer" else 1)
    # Churn histogram
    plt.figure(figsize=(20,10))
    dataframe['Churn'].hist()
    plt.savefig(os.path.join(save_path, 'churn_hist.png'))

    # Marital status histogram
    plt.figure(figsize=(20,10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(save_path, 'marital_status_hist.png'))

    # Heatmap
    plt.figure(figsize=(20,10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig(os.path.join(save_path, 'heatmap.png'))

    # Total_Trans_Ct
    plt.figure(figsize=(20,10))
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join(save_path, 'total_trans_ct.png'))


def encoder_helper(dataframe, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
                      used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    if response is None:
        new_col_name_list = [col_name + '_Churn' for col_name in category_lst]
    for col_name, new_col_name in zip(category_lst, new_col_name_list):
        col_lst = []
        col_groups = dataframe.groupby(col_name).mean()['Churn']
        for val in dataframe[col_name]:
            col_lst.append(col_groups.loc[val])
        dataframe[new_col_name] = col_lst

    return dataframe


def perform_feature_engineering(dataframe, response=None):
    '''
    input:
              dataframe: pandas dataframe
              response: string of response name [optional argument that could be
                        used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    if response is None:
        response = 'Churn'
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']
    y_full = dataframe[response]
    x_full = pd.DataFrame()
    x_full[keep_cols] = dataframe[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size= 0.3,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_test,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_test:  test response values
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    save_path = "images/results/"

    # Plot LR
    plt.figure(figsize=(15, 8))
    fpr, tpr, _ = roc_curve(y_test, y_test_preds_lr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr,
                                      estimator_name='Logistic Regression')
    display.plot()
    plt.savefig(os.path.join(save_path, 'logistic_regression_roc_curve.png'))

    # Plot RF
    plt.figure(figsize=(15, 8))
    fpr, tpr, _ = roc_curve(y_test, y_test_preds_rf)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr,
                                      estimator_name='Random Forest')
    display.plot()
    plt.savefig(os.path.join(save_path, 'random_forest_roc_curve.png'))


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # TreeExplainer
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(x_data)
    shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)
    plt.savefig(os.path.join(output_pth, 'feature_importance_bar.png'))

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(os.path.join(output_pth, 'feature_importance_ranked.png'))

def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    save_path = "models/"
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    # Results
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_test_preds_lr = lrc.predict(x_test)
    classification_report_image(y_test,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Feature importance
    feature_importance_plot(cv_rfc, x_test, "images/results/")

    # Save best model
    joblib.dump(cv_rfc.best_estimator_, os.path.join(save_path, 'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(save_path, 'logistic_model.pkl'))


if __name__ == "__main__":

    # Defs
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
    ]

    # Read data
    PTH = r"./data/bank_data.csv"
    df = import_data(PTH)
    print(df.head())

    # Eda
    perform_eda(df)

    # Encoding categorical variables
    df = encoder_helper(df, cat_columns)

    # Feature engineering
    X_train_df, X_test_df, y_train_df, y_test_df = perform_feature_engineering(df)

    # Train models
    train_models(X_train_df, X_test_df, y_train_df, y_test_df)
