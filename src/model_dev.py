import logging
from abc import ABC, abstractclassmethod

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractclassmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
        
class RandomForestModel(Model):
    """
    Random Forest Regressor model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        try:
            reg = RandomForestRegressor()
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e
