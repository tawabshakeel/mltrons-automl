from libraries import *
from utilities.utilty import Utility
from automl.mltrons_model_details import MltronsModelsDetails
from handling_data.handling_data import HandlingData


class MltronsAutoml(object):
    def __init__(self, problem_type, target_variable, order_of_features, max_models=7):
        self.target_variable = target_variable
        self.problem_type = problem_type
        self.max_models = max_models
        self.model_names = []
        self.models = []
        self.model_depth = [6, 7, 8, 9, 10, 11, 12]
        self.train_pool = None
        self.test_pool = None
        self.order_of_features = order_of_features
        self.model_explanation = MltronsModelsDetails(self.problem_type, self.order_of_features)
        self.init_models()

    def create_model_names(self):
        """
        creating cat-boost unique model_name
        """
        for i in range(self.max_models):
            self.model_names.append("Catboost_mltronsautoml_" + str(Utility.random_string_generate(7)) + "_" + str(i))

    def init_models(self):
        self.create_model_names()
        if self.problem_type == 'Regression':
            for idx, dep in enumerate(self.model_depth):
                result = CatBoostRegressor(iterations=5000,
                                           depth=dep,
                                           eval_metric=self.model_explanation.get_metric(self.problem_type),
                                           train_dir=self.model_names[idx])
                self.models.append(result)

        else:
            for idx, dep in enumerate(self.model_depth):
                result = CatBoostClassifier(iterations=5000,
                                            depth=dep,
                                            eval_metric=self.model_explanation.get_metric(self.problem_type),
                                            train_dir=self.model_names[idx])
                self.models.append(result)

    def fit(self, train_pool, test_pool):
        """

        """
        self.train_pool = train_pool
        self.test_pool = test_pool
        for idx, model in enumerate(range(self.max_models)):
            self.models[idx].fit(train_pool, eval_set=test_pool, early_stopping_rounds=100, plot=True)
            self.model_explanation.calculate_model_details(self.models[idx], test_pool)
