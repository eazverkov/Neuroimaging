from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

class Domain2Var2Model():
    def __init__(self):
        # read data
        self.model = LGBMRegressor()
        fnc = pd.read_csv(r'C:\Users\Евгений\Kaggle\TReNDS Neuroimaging\fnc.csv')
        # print(fnc.head())
        loading = pd.read_csv(r'C:\Users\Евгений\Kaggle\TReNDS Neuroimaging\loading.csv')
        ss = pd.read_csv(r'C:\Users\Евгений\Kaggle\TReNDS Neuroimaging\sample_submission.csv')
        self.train_scores = pd.read_csv(r'C:\Users\Евгений\Kaggle\TReNDS Neuroimaging\train_scores.csv')
        self.data = pd.merge(fnc, loading, on='Id', how='left')  # Все ИКСЫ, для обучения и тестов.
        # self.data = loading.copy()
        self.test_id = pd.DataFrame(list(set(ss['Id'].apply(lambda x: int(x.split('_')[0])))), columns=['Id'])  # Id тестовых данных
        self.data_X_test = pd.merge(self.data, self.test_id, on='Id', how='left')
        self.features = list(self.data.columns[1:])
        # self.features = list(loading.columns[1:])
        print("self.features: ", self.features)

    def get_x_y(self):
        y_train = self.train_scores[['Id', 'domain2_var2']]
        y_train = y_train.loc[ pd.isnull(y_train['domain2_var2'])==False , :].reset_index(drop=True)
        X_train = pd.merge(self.data, y_train, on='Id', how='right').drop(['domain2_var2'], axis=1).reset_index(drop=True)

        #

        print(X_train.head())
        print(X_train.shape)
        return X_train, y_train

    def get_x_y_test(self):
        X_test = pd.merge(self.data, self.test_id, on='Id', how='right')
        return X_test


    def cross_validation(self):
        kf = KFold(n_splits=5)
        X, y = self.get_x_y()
        print(y)
        for train_index, test_index in kf.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.loc[train_index, self.features], X.loc[test_index, self.features]
            y_train, y_test = y.loc[train_index, 'domain2_var2'], y.loc[test_index, 'domain2_var2']
            model_for_test = LGBMRegressor(n_estimators=300, max_depth=4)
            model_for_test.fit(X_train, y_train)
            test_prediction = model_for_test.predict(X_test)
            mae = mean_absolute_error(y_true=y_test, y_pred=test_prediction)
            average_pred = np.mean(test_prediction)
            metric = mae / average_pred
            print('mae:', mae)
            print("metric:", metric)
            #todo save metrics in file

  #  return test_prediction, mae, average_pred, metric

    def predict_for_test(self):
        # self.test_prediction = self.model.predict(X_test)
        X, y = self.get_x_y()
        model = LGBMRegressor(n_estimators=300, max_depth=4)
        model.fit(X.loc[ : , self.features],  y.loc[:, 'domain2_var2'])

        X_test  = self.get_x_y_test()
        test_prediction = model.predict(X_test.loc[ : , self.features])


        return test_prediction

    # def test_X(self):


if __name__ == "__main__":
    Domain2_var2_model = Domain2Var2Model()
    Domain2_var2_model.cross_validation()
    Domain2_var2_model.predict_for_test()
