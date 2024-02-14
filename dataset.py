import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def make_dataset():
    path = os.path.abspath(os.getcwd())
    filename = 'Data-train.csv'
    full_path = path + '/' + filename
    data = pd.read_csv(full_path, index_col=0)
    sales = data.sales
    data.drop('sales', axis=1, inplace=True)

    # Do one-hot encoding for the categorical variables and create the df
    OHE = OneHotEncoder(sparse=False)
    categorical_list = ["productID", "brandID", "weekday"]
    categorical_data = OHE.fit_transform(data[categorical_list])
    categorical_df = pd.DataFrame(data=categorical_data, columns=OHE.get_feature_names_out())

    # Do scaling for the non-categorical data
    ss = StandardScaler()
    continuous_data = data.drop(columns=categorical_list, axis=1)
    continuous_df = pd.DataFrame(ss.fit_transform(continuous_data), columns=continuous_data.columns)

    # Combining the categorical and continuous dataframes
    frames = [categorical_df, continuous_df]
    data_scaled = pd.concat(frames, axis=1)

    # Creating the scaled and not scaled data
    not_scaled_df = pd.concat([categorical_df, data, sales], axis=1)
    not_scaled_df.drop(["productID", "brandID", "weekday"], axis=1, inplace=True)
    scaled_df = pd.concat([data_scaled, sales], axis=1)
    data_train = pd.read_csv(full_path, index_col=0)

    return ss, not_scaled_df, scaled_df, data_train

# Function for predicted profit of currrent inventory decision
def Predicted_Profit(D,y):
    p, c, s = 20, 12, 8
    return (p-s) * pd.concat([D, y], axis=1).apply(lambda x: min(x[0], x[1]), axis=1) - (c-s) * y