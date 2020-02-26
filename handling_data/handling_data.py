from libraries import *


class HandlingData(object):
    def __init__(self):
        pass

    @staticmethod
    def split_time(df, date):
        date = date
        file = df.copy()
        temp = pd.to_datetime(file[date], errors='coerce')
        file[date + '_year'] = temp.dt.year
        file[date + '_month'] = temp.dt.month
        file[date + '_day'] = temp.dt.day
        file[date + '_dayofweek'] = temp.dt.dayofweek

        file[date + '_hour'] = temp.dt.hour
        file[date + '_minute'] = temp.dt.minute
        file[date + '_second'] = temp.dt.second
        file = file.drop(date, axis=1)

        return file

    @staticmethod
    def find_object_variable(df):

        """
        This function finds all variables that contains strings.

        Input: Dask Dataframe
        Output: list of variables that are categorical
        """

        # important data
        columns = list(df.columns)

        # get object type
        df2 = dask.datasets.timeseries()
        sample_types = list(df2.dtypes)
        object_type = sample_types[1]

        # All variable types
        list_types = df.dtypes.to_list()

        # Find all object variables
        object_variables = []
        object_variabels_index = []
        j = 0
        for i in range(len(list_types)):
            if list_types[i] == object_type:
                object_variables.append(columns[i])
                object_variabels_index.append(j)
            j = j + 1

        return object_variables, object_variabels_index

    def split_all_time_variable(self, df):
        object_variables = self.find_object_variable(df)
        df100 = df.loc[0:100]

        print("string")
        print(object_variables)

        time_variable = []

        for v in object_variables:
            try:
                file_temp = self.split_time(df100, v)

                des = file_temp.compute().describe()
                count = des[v + '_year'][0]
                print('count= ', count)

                # update count variable using summary.

                if count > 60:
                    file2 = self.split_time(df, v)
                    file2.compute()
                    time_variable.append(v)
                    df = file2
                    file2 = None
                    print("This is a time Variable: ", v)
            except Exception as e:
                print(e)
        return df, time_variable

    @classmethod
    def convert_y_variable_to_double(self, df, y_variable):
        """

        """
        df = df.compute()
        if y_variable in df.columns:
            df[y_variable] = df[y_variable].astype(str).str.replace(r'[^\d.]+', '')
            df.dropna(subset=[y_variable], axis=0, inplace=True)
            print("len", len(df))
            df = self.remove_null_rows(df, y_variable)
            print("len2", len(df))
            df[y_variable] = df[y_variable].astype(float)
        df = dd.from_pandas(df, npartitions=1)
        return df

    @staticmethod
    def converting_encoding(df):

        for i in df.columns:
            print("inside encoding")
            df[i] = df[i].map(lambda x: str(x).encode('unicode-escape').decode('utf-8'))
        print(df)
        return df

    @staticmethod
    def remove_null_rows(df, y_variable):
        df = df[(df[y_variable] != 'NULL') &
                (df[y_variable] != 'N/A') &
                (df[y_variable] != 'null') &
                (df[y_variable] != 'Null') &
                (df[y_variable] != 'None') &
                (df[y_variable] != 'NONE') &
                (df[y_variable] != 'nan') &
                (df[y_variable] != 'Nan') &
                (df[y_variable] != 'NAN') &
                (df[y_variable] != 'NaN') &
                (df[y_variable] != 'none') &
                (df[y_variable] != '') &
                (df[y_variable] != 'NA') &
                (df[y_variable] != 'na') &
                (df[y_variable].notnull())]
        return df
