"""
This is the churn_library.py module.
Artifact produced will be in images, logs and models folders.
Usage:
1. EDA
2. Feature engineering
3. Model training and saving
4. saving Result
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dataframe_image as dfi

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

sns.set()

for directory in ["logs", "images/eda", "images/results", "./models"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    # read pandas dataframe from csv file path
    data = pd.read_csv(pth)
    #
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return data


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    column_names = ["Churn", "Customer_Age", "Marital_Status", "Total_Trans"]
    for column_name in column_names:
        plt.figure(figsize=(20, 10))
        if column_name == "Churn":
            df["Churn"].hist()
        elif column_name == "Customer_Age":
            df["Customer_Age"].hist()
        elif column_name == "Marital_Status":
            df["Marital_Status"].value_counts("normalize").plot(kind="bar")
        elif column_name == "Total_Trans":
            sns.displot(df['Total_Trans_Ct'])
        elif column_name == "Heatmap":
            sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
        plt.savefig("images/eda/%s.jpg" % column_name)
        plt.close()

    # plot heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig("images/eda/Heatmap.jpg")
    plt.close()


def encoder_helper(df, category_lst):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    for category_name in category_lst:
        category_lst = []
        gender_groups = df.groupby(category_name).mean()['Churn']

        for val in df[category_name]:
            category_lst.append(gender_groups.loc[val])

        df[f"{category_name}_Churn"] = category_lst

    return df


def perform_feature_engineering(df):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    df_y = df["Churn"]
    df_x = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    df_x[keep_cols] = df[keep_cols]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        df_x, df_y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """

    classification_reports_data = {
        "Random_Forest": {
            "Train": [y_test, y_test_preds_rf],
            "Test": [y_train, y_train_preds_rf]
        },
        "Logistic_Regression": {
            "Train": [y_train, y_train_preds_lr],
            "Test": [y_test, y_test_preds_lr]
        }
    }

    for title, classification_data in classification_reports_data.items():
        for mode, data in classification_data.items():
            clf_report = pd.DataFrame(classification_report(
                data[0], data[1], output_dict=True))
            dfi.export(clf_report, f"images/results/{title}_{mode}.png")


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(f"images/{output_pth}/Feature_Importance.jpg")
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    """0
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=1000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"]
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc, x_train, "results")

    joblib.dump(cv_rfc.best_estimator_, "models/rfc_model.pkl")
    joblib.dump(lrc, "models/logistic_model.pkl")


if __name__ == "__main__":

    data_df = import_data("data/bank_data.csv")
    perform_eda(data_df)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    encoded_data_df = encoder_helper(data_df,
                                     cat_columns)
    x_train_, x_test_, y_train_, y_test_ = perform_feature_engineering(
        encoded_data_df)
    train_models(x_train_, x_test_, y_train_, y_test_)
