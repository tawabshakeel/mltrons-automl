import dask.dataframe as dd
from handling_data.handling_data import HandlingData
h = HandlingData()
ddf = dd.read_csv("titanic.csv")
ddf_transform1, lst = h.split_all_time_variable(ddf)

# step 2
ddf_transform2 =h.converting_encoding(ddf_transform1)
# print(ddf_transform2.compute())

# step 3
ddf_transform3 = h.balance_dataframe_classes(ddf_transform2)
print(ddf_transform3.compute())
