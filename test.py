import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt


# helper function for reading datatset
def read_data(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")  # convert it to datatime
    return df


df_train = read_data("data/kospi_train.csv")
df_test = read_data("data/kospi_test.csv")

max_test_day_before = 10

day_before_mse_list = []
for prev_day in range(1, max_test_day_before + 1, 1):
    X_train = df_train.drop(columns=["Date"]).shift(1).dropna().reset_index(drop=True)
    if prev_day > 0:
        for i in range(1, prev_day + 1, 1):
            X_train[f"Open-{i}"] = X_train["Open"].shift(i)
            X_train[f"Low-{i}"] = X_train["Low"].shift(i)
            X_train[f"High-{i}"] = X_train["High"].shift(i)
            X_train[f"Close-{i}"] = X_train["Close"].shift(i)
            X_train[f"Volume-{i}"] = X_train["Volume"].shift(i)
    X_train = X_train.dropna().reset_index(drop=True)

    y_train = df_train[["Close"]][prev_day + 1 :].reset_index(drop=True)

    kfold = KFold(n_splits=10)

    mse_list = []
    for train_index, test_index in kfold.split(X_train, y_train):
        X_train_fold, X_validate_fold = (
            X_train.iloc[train_index],
            X_train.iloc[test_index],
        )
        y_train_fold, y_validate_fold = (
            y_train.iloc[train_index],
            y_train.iloc[test_index],
        )

        model = LinearRegression()
        model.fit(X_train_fold, y_train_fold)

        y_pred = model.predict(X_validate_fold)
        mse = mean_squared_error(y_validate_fold, y_pred)
        mse_list.append(mse)

    day_before_mse_list.append(sum(mse_list) / len(mse_list))

plt.plot(day_before_mse_list)
plt.show()
