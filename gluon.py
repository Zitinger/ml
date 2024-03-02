# !pip install autogluon pandas numpy
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")



# from autogluon.tabular import TabularDataset, TabularPredictor
# predictor = TabularPredictor(label='target', problem_type='regression', eval_metric='symmetric_mean_absolute_percentage_error') # обучение
# predictor.fit(train, time_limit=3600)

submission = pd.DataFrame(test['id'], columns=["id"])
submission['target'] = predictor.predict(test)
submission1 = submission.sample(30000)
submission1['target'] = abs(submission1['target'])
submission1.to_csv("submission1.csv", index=False)
