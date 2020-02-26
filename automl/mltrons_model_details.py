from libraries import *


class MltronsModelsDetails(object):
    def __init__(self, problem_type, order_of_features):
        self.model_details = dict()
        self.problem_type = problem_type
        self.order_of_features = order_of_features

    @staticmethod
    def get_metric(problem_type):
        """
        TYPE OF PROBLEM
        """
        if problem_type != 'Regression':
            metric = "Accuracy"
        else:
            metric = 'RMSE'
        return metric

    def calculate_model_details(self, model, test_pool):

        if self.problem_type == 'Classification':

            # score
            score = self.model_score(model, test_pool)
            self.model_details["Metric"] = {"name": ["Accuracy"], "value": [score]}
            col, values = self.feature_importance(model)

            graph = dict()
            graph["Variable Importance"] = {"variables": col, "importance": values}

            conf_mat = self.confusion_matrix(model, test_pool)
            graph["confusion_matrix"] = conf_mat
            self.model_details["graph"] = graph

        else:

            graph = dict()
            score = self.model_score(model, test_pool)
            self.model_details["Metric"] = {"name": ["MAPE"], "value": [score]}
            col, values = self.feature_importance(model)
            graph["Variable Importance"] = {"variables": col, "importance": values}
            self.model_details["graph"] = graph

        return self.model_details

    @staticmethod
    def model_score(model, test_pool):
        data = model.score(test_pool)
        return data

    def feature_importance(self, model):
        importance = list(model.get_feature_importance())
        df = pd.DataFrame()
        df['columns'] = self.order_of_features
        df['importance'] = importance

        col = list(df['columns'])
        values = list(df['importance'])
        return col, values

    @staticmethod
    def confusion_matrix(model, test_pool):
        """
        Confusion Matrix is only for Classification
        """

        c_matrix = get_confusion_matrix(model, test_pool)
        df = pd.DataFrame(c_matrix, columns=model.classes_, index=model.classes_)
        res = json.loads(df.to_json(orient='index'))
        return res
