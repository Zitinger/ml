# !pip install mljar-supervised
# !pip install matplotlib

import numpy as np
import pandas as pd
from supervised import AutoML

df = pd.read_csv("data.csv")

for i in df.columns[:-1]:
  df[i].fillna(df[i].mean(), inplace=True)
  
train = df[df['target'] != '?']
testX = df.drop(['id', 'target'], axis=1).to_numpy()

# automl = AutoML(mode="Compete", results_path='Automl_777', total_time_limit=7200) # обучение
# automl.fit(train.iloc[:,:-1], train['target']) # обучение

automl = AutoML(results_path='Automl_777')
pred = automl.predict(df.iloc[:,:-1])
res = pd.DataFrame({ 'id' : df.id, 'target' : pred})
res.to_csv('catboost2.csv', index=False)