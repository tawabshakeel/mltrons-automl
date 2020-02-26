import sys

import dask.dataframe as dd
from handling_data.handling_data import HandlingData
from automl.mltrons_automl import MltronsAutoml

ddf = dd.read_csv("titanic.csv")
target_variable = 'Survived'
problem_type = 'Classification'

h = HandlingData(ddf, target_variable, problem_type)
train_pool, test_pool, order_of_features = h.init_data_handling()
print(order_of_features)
sys.exit()
auto_ml = MltronsAutoml(problem_type, target_variable, order_of_features)
auto_ml.fit(train_pool, test_pool)

### list of models
#auto_ml.models

### getting prediction
auto_ml.models[0].predict(test_pool)

###
