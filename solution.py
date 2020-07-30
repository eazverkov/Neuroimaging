import pandas as pd
import numpy as np
from datetime import datetime, date, time

from models.age_model import AgeModel
from models.domain1_var1_model import Domain1Var1Model
from models.domain1_var2_model import Domain1Var2Model
from models.domain2_var1_model import Domain2Var1Model
from models.domain2_var2_model import Domain2Var2Model


class Solution():
    def __init__(self):
        """
        create test  df with zeros (Id, age, ...)

        """
        self.ss = pd.read_csv('sample_submission.csv')
        print(self.ss)

        test_id = pd.DataFrame(list(set(self.ss['Id'].apply(lambda x: int(x.split('_')[0])))),columns=['Id'])  # Id тестовых данных
        df_with_answers = pd.DataFrame(test_id['Id'])
        #df_with_answers = pd.DataFrame({'Id': [], 'age': [], 'domain1_var1': [], 'domain1_var2': [], 'domain2_var1': [], 'domain2_var2': []})
        df_with_answers['age'] = df_with_answers.apply(lambda _: 0, axis=1)
        df_with_answers['domain1_var1'] = df_with_answers.apply(lambda _: 0, axis=1)
        df_with_answers['domain1_var2'] = df_with_answers.apply(lambda _: 0, axis=1)
        df_with_answers['domain2_var1'] = df_with_answers.apply(lambda _: 0, axis=1)
        df_with_answers['domain2_var2'] = df_with_answers.apply(lambda _: 0, axis=1)
        self.df_with_answers = df_with_answers.copy()

    def run(self):

        age_model = AgeModel()
        age_prediction = age_model.predict_for_test()

        domain1_var1_model = Domain1Var1Model()
        domain1_var1_model_prediction = domain1_var1_model.predict_for_test()

        domain1_var2_model = Domain1Var2Model()
        domain1_var2_model_prediction = domain1_var2_model.predict_for_test()

        domain2_var1_model = Domain2Var1Model()
        domain2_var1_model_prediction = domain2_var1_model.predict_for_test()

        domain2_var2_model = Domain2Var2Model()
        domain2_var2_model_prediction = domain2_var2_model.predict_for_test()

        print("age_prediction.shape: ", age_prediction.shape)
        print("self.df_with_answers.shape: ",self.df_with_answers.shape)
        self.df_with_answers['age'] = age_prediction
        self.df_with_answers['domain1_var1'] = domain1_var1_model_prediction
        self.df_with_answers['domain1_var2'] = domain1_var2_model_prediction
        self.df_with_answers['domain2_var1'] = domain2_var1_model_prediction
        self.df_with_answers['domain2_var2'] = domain2_var2_model_prediction
        # положить все в датафрейм с ответами

    def save_in_submission_format(self):
        # generate name with datetime
        x, y = [], []
        for i in ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']:
            x += list(self.df_with_answers['Id'].apply(lambda x: str(x) + '_' + i))
            y += list(self.df_with_answers[i]) #???????????????????????????????????????????????????????????????
        dfansw = pd.DataFrame({'Id': x, 'Predicted': y})
        t = str(datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
        # pd merge to sample_submission
        print(dfansw)
        output = pd.merge(self.ss[['Id']], dfansw, how='left', on='Id')
        print(output)
        output.to_csv('submits/submit_{}.csv'.format(t), index=None)


if __name__ == "__main__":
    solution = Solution()
    solution.run()
    solution.save_in_submission_format()
