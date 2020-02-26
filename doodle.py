import dask.dataframe as dd
from handling_data.handling_data import HandlingData
from automl.mltrons_automl import MltronsAutoml

ddf = dd.read_csv("titanic.csv")
target_variable = 'Survived'

ml_auto = MltronsAutoml("Classification", target_variable)
h = HandlingData(ddf,target_variable)

# step 1

ddf_transform1, lst = h.split_all_time_variable(ddf)

# step 2
ddf_transform2 = h.converting_encoding(ddf_transform1)

# step 3
ddf_transform3 = h.making_target_column_at_end(ddf_transform2, target_variable)

# step 4
order_of_variables_lst = h.order_of_columns(ddf_transform3, target_variable)

# step 5
ddf_transform3 = h.balance_dataframe_classes(ddf_transform3, target_variable, order_of_variables_lst[0])

# step 6
train_path, test_path, train_df = h.train_test_csv(ddf_transform3)

# step 7
cd_path, cat_index, cat_names = h.create_cd_catboost(train_df, target_variable)

# step 8
train_pool = h.make_pool(train_path, cd_path)
test_pool = h.make_pool(test_path, cd_path)

# step 9
ml_auto.fit(train_pool, test_pool)

print(ml_auto)
