from zenml.steps import BaseParameters
from typing import Literal

class ModelNameConfig(BaseParameters):
    """ Model Configs. """
    model_name: Literal["LinearRegression", "RandomForestRegressor"] = "LinearRegression"
