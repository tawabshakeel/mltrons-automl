from libraries import *


class HandlingData(object):
    def __init__(self):
        pass

    @staticmethod
    def order_of_columns(df, target_feature):
        columns_lst = list(df.columns)
        columns_lst.remove(target_feature)
        return columns_lst

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
        print("object variables", object_variables)
        time_variable = []

        for variables in object_variables:
            try:
                file_temp = self.split_time(df100, variables)

                res = file_temp.compute().describe()
                count = res[variables + '_year'][0]
                print('count= ', count)
                if count > 60:
                    df_temp = self.split_time(df, variables)
                    df_temp.compute()
                    time_variable.append(variables)
                    df = df_temp
                    print("This is a time Variable: ", variables)
            except Exception as e:
                print(e)
        return df, time_variable

    @classmethod
    def convert_y_variable_to_double(cls, df, y_variable):
        """

        """
        df = df.compute()
        if y_variable in df.columns:
            df[y_variable] = df[y_variable].astype(str).str.replace(r'[^\d.]+', '')
            df.dropna(subset=[y_variable], axis=0, inplace=True)
            df = cls.remove_null_rows(df, y_variable)
            df[y_variable] = df[y_variable].astype(float)
        df = dd.from_pandas(df, npartitions=1)
        return df

    @staticmethod
    def converting_encoding(df):

        for i in df.columns:
            df[i] = df[i].map(lambda x: str(x).encode('unicode-escape').decode('utf-8'))
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

    @staticmethod
    def balance_dataset_details(df, target_variable, variable):

        step1 = df.groupby(target_variable)[[variable]].count().compute()

        step1.reset_index(inplace=True)
        step1.rename(columns={variable: "value"}, inplace=True)

        majority_class = step1[step1['value'] == step1['value'].max()].to_json(orient="records")
        maj_class_value = json.loads(majority_class)[0]['value']
        maj_class_name = json.loads(majority_class)[0][target_variable]

        other_classes = []
        for c in json.loads(step1.to_json(orient="records")):
            if c[target_variable] != maj_class_name:
                other_classes.append((c[target_variable], c['value']))

        return maj_class_value, maj_class_name, other_classes

    def balance_dataframe_classes(self, df, target_variable, variable):

        maj_value, maj_name, other_classes = self.balance_dataset_details(df, target_variable, variable)

        df_majority = df[df[target_variable] == maj_name]

        for c_name in other_classes:
            # Upsample minority class
            df_minority = df[df[target_variable] == c_name[0]]

            df_minority_upsampled = df_minority.sample(frac=(maj_value - c_name[1]) / (c_name[1]), replace=True)

            df_majority = dd.concat([df_majority, df_minority, df_minority_upsampled])

        df_majority = df_majority.reset_index()

        return df_majority

    def make_pool(self, df, target_variable):

        cat_variable, cat_variables_index = self.find_object_variable(df)

        col = list(df.columns)
        label = col.index(target_variable)
        print("label", label)
        try:
            cat_variables_index.remove(label)
        except:
            print("Target variable in an integer.")

        feature_names = dict()

        for i in range(len(col)):
            if i == label:
                continue
            feature_names[i] = col[i]

        return label, cat_variables_index, feature_names, cat_variable

    def train_test_csv(self, df):
        # if "index" not in df.columns:
        #     df["index"] = 1
        #     df['index'] = df['index'].cumsum()
        # train, test = self.train_test_split(df)

        train, test = df.random_split([0.85, 0.15])
        train_file_name = 'media/User/Datasets/' + str(time.time()) + '_train.csv'
        test_file_name = 'media/User/Datasets/' + str(time.time()) + '_test.csv'

        train.to_csv(
            train_file_name,
            index=False, sep=',', header=True, single_file=True
        )
        test.to_csv(
            test_file_name,
            index=False, sep=',', header=True, single_file=True
        )

        return train_file_name, test_file_name, train

    def create_cd_catboost(self, df, target_variable):

        label, cat_variables_index, feature_names ,cat_variable_names= self.make_pool(df, target_variable)

        path = 'media/User/Datasets/' + str(time.time()) + '_train.cd'
        create_cd(
            label=label,
            cat_features=cat_variables_index,
            feature_names=feature_names,
            output_path=path
        )
        return path, cat_variables_index,cat_variable_names