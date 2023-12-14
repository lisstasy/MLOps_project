import logging
from typing import Tuple
from typing_extensions import Annotated

import pandas as pd
from zenml import step
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin

@step
def evaluate_model (model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
    ) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"]
]:
    """
    Evaluates the model on the ingested data.

    Args:
        model: RegressorMixin
        X_test: Testing data 
        y_test: Testing labels
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2_score = r2_class.calculate_scores(y_test, prediction)

        return rmse, r2_score

    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e

